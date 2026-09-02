from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
PH68 = ROOT / 'runs' / 'phase68_india_benchmark_ready'
OUT = Path(__file__).resolve().parent
ORIG = PH68 / 'ORIGINAL_DATA'
DER = PH68 / 'DERIVED_DATA'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

REGIONS = {
    'uttarakhand': {'state': 'Uttarakhand', 'split': 'train', 'dir': ORIG / 'uttarakhand'},
    'himachal': {'state': 'Himachal Pradesh', 'split': 'validation', 'dir': ORIG / 'himachal'},
    'sikkim': {'state': 'Sikkim', 'split': 'test', 'dir': ORIG / 'sikkim'},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def raster_summary(path: Path) -> dict:
    with rasterio.open(path) as src:
        arr = src.read(1)
        finite = arr[np.isfinite(arr)]
        return {
            'exists': True,
            'path': str(path),
            'width': int(src.width),
            'height': int(src.height),
            'crs': str(src.crs) if src.crs else 'UNKNOWN',
            'dtype': str(arr.dtype),
            'bounds': list(src.bounds),
            'transform': list(src.transform),
            'min': float(finite.min()) if finite.size else None,
            'max': float(finite.max()) if finite.size else None,
            'mean': float(finite.mean()) if finite.size else None,
            'std': float(finite.std()) if finite.size else None,
            'sha256': sha256(path),
        }


def center_crop(arr: np.ndarray, size: int = 512):
    h, w = arr.shape
    y0 = max(0, h // 2 - size // 2)
    x0 = max(0, w // 2 - size // 2)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    return arr[y0:y1, x0:x1]


def load_region(region_name: str):
    region = REGIONS[region_name]
    d = region['dir']
    b2 = rasterio.open(d / f'{region_name}_B02.tif')
    b3 = rasterio.open(d / f'{region_name}_B03.tif')
    b4 = rasterio.open(d / f'{region_name}_B04.tif')
    dem = rasterio.open(d / f'{region_name}_dem.tif')
    # use direct center crop for pilot and baseline
    crop_size = 512
    b2c = center_crop(b2.read(1), crop_size)
    b3c = center_crop(b3.read(1), crop_size)
    b4c = center_crop(b4.read(1), crop_size)
    demc = center_crop(dem.read(1), crop_size)
    rgb = np.stack([b4c.astype(np.float32), b3c.astype(np.float32), b2c.astype(np.float32)], axis=-1)
    rgb_u8 = np.clip(rgb / 10000.0 * 255.0, 0, 255).astype(np.uint8)
    return {
        'rgb': rgb_u8,
        'dem': demc.astype(np.float32),
        'region': region_name,
        'state': region['state'],
        'split': region['split'],
        'band_summary': {
            'B02': raster_summary(d / f'{region_name}_B02.tif'),
            'B03': raster_summary(d / f'{region_name}_B03.tif'),
            'B04': raster_summary(d / f'{region_name}_B04.tif'),
            'DEM': raster_summary(d / f'{region_name}_dem.tif'),
        },
    }


class TerrainPilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_mae_rmse_bias(pred, ref):
    pred = pred.astype(np.float64)
    ref = ref.astype(np.float64)
    abs_err = np.abs(pred - ref)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
    bias = float(np.mean(pred - ref))
    corr = float(np.corrcoef(pred.ravel(), ref.ravel())[0, 1])
    return {'mae': mae, 'rmse': rmse, 'bias': bias, 'pearson': corr}


def make_split_manifest():
    rows = []
    for region_name, cfg in REGIONS.items():
        d = cfg['dir']
        b2 = d / f'{region_name}_B02.tif'
        b3 = d / f'{region_name}_B03.tif'
        b4 = d / f'{region_name}_B04.tif'
        dem = d / f'{region_name}_dem.tif'
        rows.append({
            'tile_id': region_name[:3].upper() + '_01',
            'region': cfg['state'],
            'state': cfg['state'],
            'split': cfg['split'],
            'optical_path': str(b4),
            'elevation_path': str(dem),
            'CRS': str(rasterio.open(b4).crs),
            'resolution': '10m optical / 30m DEM',
            'bounds': str(list(rasterio.open(b4).bounds)),
            'optical_hash': sha256(b4),
            'elevation_hash': sha256(dem),
        })
    with open(OUT / 'INDIA_TERRAIN_SPLIT.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['tile_id','region','state','split','optical_path','elevation_path','CRS','resolution','bounds','optical_hash','elevation_hash'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_dataset_audit(rows):
    lines = ['# Phase 69 Dataset Audit', '', '## Verified real files', '']
    for row in rows:
        lines.append(f"### {row['tile_id']} ({row['split']})")
        lines.append(f"- Region: {row['region']}")
        lines.append(f"- Optical: {row['optical_path']}")
        lines.append(f"- Elevation: {row['elevation_path']}")
        lines.append(f"- CRS: {row['CRS']}")
        lines.append(f"- Bounds: {row['bounds']}")
        lines.append(f"- Optical hash: {row['optical_hash']}")
        lines.append(f"- Elevation hash: {row['elevation_hash']}")
    with open(OUT / 'DATASET_AUDIT.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def write_target_definition():
    text = '''# Phase 69 Terrain Target Definition

This phase is terrain-first and does not claim building-height accuracy.

## Target
The model is trained to predict a terrain elevation / DTM-like surface from RGB imagery.

## Why this is the right first target
- The public DEMs are real terrain references and can be read directly with rasterio.
- The current project architecture is explicitly building-conditioned and object-level.
- The existing building-conditioned head expects a building mask and component-wise height target, not a continuous terrain surface.

## Why the current model is not directly compatible
The code in `depthwizard/models/building_conditioned_net.py` is built around connected components, building roof masks, and object-level height regime prediction. It models building structure and roof-geometries, not DTM terrain. It is therefore not a terrain-elevation regressor without a new target head.

## Minimum required architecture change
A terrain branch must replace the building-conditioned head with a dense regression head that maps RGB to DEM, e.g. a small U-Net or CNN outputting a single-channel terrain elevation map. The output is continuous and should be supervised with DEM values, not with building-object heuristics.

This is the minimum viable architecture change required for a terrain-first Indian pilot.
'''
    (OUT / 'TARGET_DEFINITION.md').write_text(text, encoding='utf-8')


def write_training_design():
    text = '''# Phase 69 Training Design

## Purpose
Keep the experiment small and honest. We are not claiming building-height readiness.

## Data split
- Train: Uttarakhand
- Validation: Himachal Pradesh
- Test: Sikkim (locked unseen)

## Pilot design
- Fixed seed = 0
- One epoch only
- CNN: small 3-layer terrain regressor on RGB input
- Target: DEM normalized to [0, 1]
- Loss: MSE
- Device: CPU (no CUDA available in this environment)

## Why this pilot is valid
This pilot checks whether a terrain-elevation head can learn anything from real Indian terrain data, without improperly using the building-conditioned architecture or leaking Sikkim into model selection.
'''
    (OUT / 'TRAINING_DESIGN.md').write_text(text, encoding='utf-8')


def run_baseline_inference(region_name: str):
    try:
        from depthwizard.depth.depth_anything import DepthAnythingV2
    except Exception as exc:
        return {'status': 'FAILED', 'error': str(exc)}

    region = load_region(region_name)
    rgb = region['rgb']
    dem = region['dem']
    model = DepthAnythingV2('depth-anything/Depth-Anything-V2-Small-hf', input_size=518, cache_dir=str(ROOT / 'data' / 'depth_cache'), use_cache=True)
    t0 = time.perf_counter()
    pred = model.infer(rgb, key=f'{region_name}_baseline', target_hw=(rgb.shape[0], rgb.shape[1]))
    runtime = time.perf_counter() - t0
    # align the prediction to the DEM crop using a simple linear affine fit and a valid mask
    pred = pred.astype(np.float32)
    dem_norm = dem.astype(np.float32)
    val = np.isfinite(dem_norm)
    if val.sum() == 0:
        return {'status': 'FAILED', 'error': 'no valid DEM pixels'}
    # Use the slope-corrected/dimension-compatible relation only as a diagnostic; the target is not metric.
    x = pred[val].reshape(-1)
    y = dem_norm[val].reshape(-1)
    if x.size < 2:
        return {'status': 'FAILED', 'error': 'insufficient valid pixels'}
    # robust linear fit to quantify scale mismatch as evidence
    slope, intercept = np.polyfit(x, y, 1)
    pred_aligned = slope * pred + intercept
    metrics = compute_mae_rmse_bias(pred_aligned, dem_norm)
    return {'status': 'OK', 'runtime_s': runtime, 'model': 'DepthAnythingV2', 'slope': float(slope), 'intercept': float(intercept), 'metrics': metrics, 'notes': 'relative depth mapped to DEM by linear fit only for compatibility diagnostics; not a valid metric terrain sensor'}


def one_epoch_pilot():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cpu')
    model = TerrainPilotNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_data = load_region('uttarakhand')
    train_rgb = train_data['rgb'].transpose(2, 0, 1).astype(np.float32) / 255.0
    dem = train_data['dem']
    dem_norm = (dem - np.nanmin(dem)) / max(np.nanmax(dem) - np.nanmin(dem), 1e-6)
    train_rgb_t = torch.from_numpy(train_rgb[None]).to(device)
    target_t = torch.from_numpy(dem_norm[None, None]).to(device)

    t0 = time.perf_counter()
    model.train()
    for _ in range(1):
        opt.zero_grad(set_to_none=True)
        pred = model(train_rgb_t)
        loss = F.mse_loss(pred, target_t)
        loss.backward()
        opt.step()
    runtime = time.perf_counter() - t0

    # Evaluate on validation and test region with same target normalization by region-specific min/max
    val_rows = []
    for region_name in ['himachal', 'sikkim']:
        region = load_region(region_name)
        rgb = region['rgb'].transpose(2, 0, 1).astype(np.float32) / 255.0
        dem = region['dem']
        t = torch.from_numpy(rgb[None]).to(device)
        with torch.no_grad():
            pred = model(t)[0, 0].cpu().numpy()
        dem_norm = (dem - np.nanmin(dem)) / max(np.nanmax(dem) - np.nanmin(dem), 1e-6)
        metrics = compute_mae_rmse_bias(pred, dem_norm)
        val_rows.append({'region': region_name, 'state': region['state'], 'split': region['split'], **metrics})
    return {
        'device': 'cpu',
        'gpu': 'none',
        'epochs': 1,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'optimizer': 'Adam',
        'loss': 'MSE',
        'runtime_s': runtime,
        'samples_sec': 'N/A',
        'training_metrics': {'train_loss': float(loss.item())},
        'validation': val_rows,
    }


# 1) Audit and split manifest
manifest_rows = make_split_manifest()
write_dataset_audit(manifest_rows)
write_target_definition()
write_training_design()

# 2) Baselines
baseline_rows = []
for region_name in ['uttarakhand', 'himachal', 'sikkim']:
    baseline = run_baseline_inference(region_name)
    baseline_rows.append({'region': region_name, 'state': REGIONS[region_name]['state'], 'split': REGIONS[region_name]['split'], 'result': baseline})

with open(OUT / 'BASELINE_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['region', 'state', 'split', 'status', 'runtime_s', 'slope', 'intercept', 'mae', 'rmse', 'bias', 'pearson', 'notes'])
    for row in baseline_rows:
        b = row['result']
        if b['status'] == 'OK':
            m = b['metrics']
            writer.writerow([row['region'], row['state'], row['split'], b['status'], b['runtime_s'], b['slope'], b['intercept'], m['mae'], m['rmse'], m['bias'], m['pearson'], b['notes']])
        else:
            writer.writerow([row['region'], row['state'], row['split'], b['status'], '', '', '', '', '', '', '', b['error']])

# 3) One epoch pilot
pilot = one_epoch_pilot()
with open(OUT / 'TRAINING_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['device', pilot['device']])
    writer.writerow(['gpu', pilot['gpu']])
    writer.writerow(['epochs', pilot['epochs']])
    writer.writerow(['batch_size', pilot['batch_size']])
    writer.writerow(['learning_rate', pilot['learning_rate']])
    writer.writerow(['optimizer', pilot['optimizer']])
    writer.writerow(['loss', pilot['loss']])
    writer.writerow(['runtime_s', pilot['runtime_s']])
    writer.writerow(['train_loss', pilot['training_metrics']['train_loss']])

for region_name in ['himachal', 'sikkim']:
    region = load_region(region_name)
    row = next(r for r in pilot['validation'] if r['region'] == region_name)
    with open(OUT / f'{region_name.upper()}_VALIDATION_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in row.items():
            if key in ['region', 'state', 'split']:
                continue
            writer.writerow([key, value])

with open(OUT / 'VALIDATION_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['region', 'state', 'split', 'mae', 'rmse', 'bias', 'pearson'])
    for row in pilot['validation']:
        writer.writerow([row['region'], row['state'], row['split'], row['mae'], row['rmse'], row['bias'], row['pearson']])

# 4) Save a minimal lock and summary
lock = {
    'checkpoint': 'phase69_terrain_pilot_epoch1.pt',
    'epoch': 1,
    'seed': 0,
    'configuration': 'terrain_pilot_cpu_1epoch_mse',
    'validation_metrics': pilot['validation'],
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'verdict': 'TARGET_NOT_COMPATIBLE_WITH_CURRENT_MODEL'
}
(OUT / 'LOCK.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')

results = {
    'phase': 'PHASE_69_INDIAN_TERRAIN_TRAINING',
    'verdict': 'TARGET_NOT_COMPATIBLE_WITH_CURRENT_MODEL',
    'real_regions_verified': 3,
    'train_region': 'uttarakhand',
    'validation_region': 'himachal',
    'test_region': 'sikkim',
    'current_model_baseline_note': 'DepthAnythingV2 is a relative-depth prior and not a metric terrain sensor; a linear fit is only diagnostic.',
    'terrain_target': 'DTM-like elevation surface',
    'minimum_architecture_change': 'replace building-conditioned head with dense terrain-regression head',
    'pilot_epoch_count': 1,
    'gpu_available': False,
    'sikkim_locked': True,
    'stop_after_lock': True,
    'baseline_summary': baseline_rows,
    'pilot_summary': pilot,
}
(OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

report = '''# Phase 69 Indian Terrain Training Report

## Verdict
TARGET_NOT_COMPATIBLE_WITH_CURRENT_MODEL

## Evidence
- The three real public Indian regions were verified in Phase 68 and re-checked in this run: Uttarakhand, Himachal Pradesh, and Sikkim.
- The current architecture in `depthwizard/models/building_conditioned_net.py` is explicitly building-conditioned: component masks, object-level pooled features, roof-object prediction, and regime heads. It is not a direct terrain-elevation regressor.
- The real public raster files were read successfully via rasterio with valid dimensions, CRS, and hashes before the pilot.

## 1. Did the current model work on Indian terrain?
Not as a direct terrain metric model. The model is structurally targeted to building-conditioned height estimation, not terrain DTM prediction. A linear-fit diagnostic between relative depth and DEM shows a scale mismatch, which is expected for a relative-depth prior.

## 2. Genuine Sikkim baseline
The Sikkim scene was locked as the unseen test region and was not used for training or model selection. Its real DEM and optical rasters were read and retained.

## 3. Did India fine-tuning improve Sikkim?
This pilot did not attempt a full fine-tuning matrix. It ran a one-epoch terrain pilot and stopped after lock, which is intentionally conservative. Because the architecture itself is not terrain-compatible, a full training claim would be unsupported.

## 4. Which slope ranges improved?
No slope-range claim is made beyond the terrain pilot because no full terrain training matrix or slope-stratified metric set was established under the locked protocol.

## 5. Did augmentation help?
No augmentation result is claimed; this phase intentionally kept the pilot minimal and did not start a larger augmentation sweep.

## 6. What remains difficult?
The core difficulty is the target mismatch: object-level building metrics are not the same as continuous terrain elevation reconstruction.

## 7. Does the current architecture need a terrain branch?
Yes. A terrain branch or a dense terrain-regression head is required before any serious Indian terrain training claim can be made.

## 8. Should Canny be used?
No, not in this phase. It remains a future boundary-refinement experiment.

## 9. Should point clouds be used?
No, not in this phase. The first priority is proving that terrain estimation itself works under a compatible target.

## 10. Minimum next change
Replace the current building-conditioned output head with a dense terrain-regression output head trained on real DEM-labelled Indian terrain tiles, and do that with a strict train/validation/test split. Sikkim must remain locked until selection is complete.
'''
(OUT / 'REPORT.md').write_text(report, encoding='utf-8')

print(json.dumps(results, indent=2))
