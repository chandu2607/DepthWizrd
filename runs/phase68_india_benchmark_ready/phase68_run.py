from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]

OUT = Path(__file__).resolve().parent
ORIG = OUT / 'ORIGINAL_DATA'
DER = OUT / 'DERIVED_DATA'
for d in [ORIG, DER]:
    d.mkdir(parents=True, exist_ok=True)

REGIONS = {
    'uttarakhand': {
        'bands': {
            'B02': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/R/MU/2026/8/S2A_44RMU_20260830_1_L2A/B02.tif',
            'B03': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/R/MU/2026/8/S2A_44RMU_20260830_1_L2A/B03.tif',
            'B04': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/R/MU/2026/8/S2A_44RMU_20260830_1_L2A/B04.tif',
        },
        'dem': 'https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N30_00_E080_00_DEM/Copernicus_DSM_COG_10_N30_00_E080_00_DEM.tif',
        'split': 'train',
        'label': 'Uttarakhand',
    },
    'himachal': {
        'bands': {
            'B02': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/EQ/2025/12/S2C_43REQ_20251227_0_L2A/B02.tif',
            'B03': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/EQ/2025/12/S2C_43REQ_20251227_0_L2A/B03.tif',
            'B04': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/EQ/2025/12/S2C_43REQ_20251227_0_L2A/B04.tif',
        },
        'dem': 'https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N30_00_E076_00_DEM/Copernicus_DSM_COG_10_N30_00_E076_00_DEM.tif',
        'split': 'validation',
        'label': 'Himachal Pradesh',
    },
    'sikkim': {
        'bands': {
            'B02': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/45/R/WL/2025/12/S2C_45RWL_20251229_0_L2A/B02.tif',
            'B03': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/45/R/WL/2025/12/S2C_45RWL_20251229_0_L2A/B03.tif',
            'B04': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/45/R/WL/2025/12/S2C_45RWL_20251229_0_L2A/B04.tif',
        },
        'dem': 'https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N27_00_E088_00_DEM/Copernicus_DSM_COG_10_N27_00_E088_00_DEM.tif',
        'split': 'test',
        'label': 'Sikkim',
    },
}


def download(url: str, path: Path) -> dict:
    if path.exists():
        status = 200
        size = path.stat().st_size
    else:
        r = requests.get(url, stream=True, timeout=120)
        status = r.status_code
        size = None
        if status == 200:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            size = path.stat().st_size
    return {'url': url, 'status': status, 'size_bytes': size, 'local_path': str(path)}


def raster_ok(path: Path) -> dict:
    if not path.exists():
        return {'exists': False}
    with rasterio.open(path) as src:
        arr = src.read(1)
        finite = arr[np.isfinite(arr)]
        return {
            'exists': True,
            'width': int(src.width),
            'height': int(src.height),
            'crs': str(src.crs) if src.crs else 'UNKNOWN',
            'dtype': str(arr.dtype),
            'min': float(finite.min()) if finite.size else None,
            'max': float(finite.max()) if finite.size else None,
            'mean': float(finite.mean()) if finite.size else None,
            'std': float(finite.std()) if finite.size else None,
        }


rows = []
for region_name, region in REGIONS.items():
    region_dir = ORIG / region_name
    region_dir.mkdir(exist_ok=True)
    for band_name, url in region['bands'].items():
        p = region_dir / f'{region_name}_{band_name}.tif'
        info = download(url, p)
        info['region'] = region_name
        info['label'] = region['label']
        info['split'] = region['split']
        info['band'] = band_name
        info['raster'] = raster_ok(p)
        rows.append(info)

    dem_path = region_dir / f'{region_name}_dem.tif'
    dem_info = download(region['dem'], dem_path)
    dem_info['region'] = region_name
    dem_info['label'] = region['label']
    dem_info['split'] = region['split']
    dem_info['band'] = 'DEM'
    dem_info['raster'] = raster_ok(dem_path)
    rows.append(dem_info)

with open(OUT / 'benchmark_manifest.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region', 'label', 'split', 'band', 'url', 'status', 'size_bytes', 'local_path', 'raster'])
    writer.writeheader()
    for row in rows:
        writer.writerow({
            'region': row['region'],
            'label': row['label'],
            'split': row['split'],
            'band': row['band'],
            'url': row['url'],
            'status': row['status'],
            'size_bytes': row['size_bytes'],
            'local_path': row['local_path'],
            'raster': json.dumps(row['raster'], sort_keys=True),
        })

# Build aligned crops for each region.
for region_name, region in REGIONS.items():
    region_dir = ORIG / region_name
    b2 = rasterio.open(region_dir / f'{region_name}_B02.tif')
    b3 = rasterio.open(region_dir / f'{region_name}_B03.tif')
    b4 = rasterio.open(region_dir / f'{region_name}_B04.tif')
    dem = rasterio.open(region_dir / f'{region_name}_dem.tif')

    cx = b2.width // 2
    cy = b2.height // 2
    crop_w = min(2048, b2.width)
    crop_h = min(2048, b2.height)
    xoff = max(0, cx - crop_w // 2)
    yoff = max(0, cy - crop_h // 2)
    window = Window(xoff, yoff, crop_w, crop_h)
    transform = b2.window_transform(window)
    b2_crop = b2.read(1, window=window)
    b3_crop = b3.read(1, window=window)
    b4_crop = b4.read(1, window=window)
    rgb = np.stack([b4_crop, b3_crop, b2_crop], axis=-1).astype(np.float32)
    rgb = np.clip(rgb / 10000.0 * 255.0, 0, 255).astype(np.uint8)

    out_rgb = DER / f'{region_name}_rgb_crop.tif'
    with rasterio.open(out_rgb, 'w', driver='GTiff', height=rgb.shape[0], width=rgb.shape[1], count=3, dtype='uint8', crs=b2.crs, transform=transform, compress='lzw') as dst:
        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)

    dem_crop = np.empty((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
    reproject(
        source=rasterio.band(dem, 1),
        destination=dem_crop,
        src_transform=dem.transform,
        src_crs=dem.crs,
        dst_transform=transform,
        dst_crs=b2.crs,
        resampling=Resampling.bilinear,
    )
    out_dem = DER / f'{region_name}_dem_aligned.tif'
    with rasterio.open(out_dem, 'w', driver='GTiff', height=dem_crop.shape[0], width=dem_crop.shape[1], count=1, dtype='float32', crs=b2.crs, transform=transform, nodata=np.nan, compress='lzw') as dst:
        dst.write(dem_crop.astype(np.float32), 1)

    print('REGION_OK', region_name, 'RGB', out_rgb, 'DEM', out_dem)

# final summary
summary = {
    'phase': 'PHASE_68_INDIAN_BENCHMARK',
    'real_regions': 3,
    'regions': [
        {'name': 'uttarakhand', 'split': 'train', 'status': 'real_public_data_verified'},
        {'name': 'himachal', 'split': 'validation', 'status': 'real_public_data_verified'},
        {'name': 'sikkim', 'split': 'test', 'status': 'real_public_data_verified'},
    ],
    'terrain_reference': 'public DEM verified via rasterio open',
    'building_reference': 'not yet acquired; coarse DEM is not validated building-height ground truth',
    'model_training_started': False,
    'current_claim': 'The benchmark is evidence-based and physically downloaded to disk. The model baseline is not retrained or modified at this stage.',
}
with open(OUT / 'phase68_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
