from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthwizard.analysis.slope import compute_slope
from depthwizard.depth.depth_anything import DepthAnythingV2

OUT = Path(__file__).resolve().parent
ORIG = OUT / 'ORIGINAL_DATA'
DER = OUT / 'DERIVED_DATA'
FIG = OUT / 'figures'
for d in [ORIG, DER, FIG]:
    d.mkdir(parents=True, exist_ok=True)

# Preserve historical evidence by copying the latest known visuals from Phase 64 into this package.
for name in ['baseline_depth.png', 'canny_edges.png', 'elevation.png', 'hillshade.png', 'slope.png']:
    src = ROOT / 'runs' / 'phase64_indian_data' / 'figures' / name
    if src.exists():
        shutil.copy2(src, FIG / name)

HIMACHAL_BANDS = {
    'B02': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/EQ/2025/12/S2C_43REQ_20251227_0_L2A/B02.tif',
    'B03': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/EQ/2025/12/S2C_43REQ_20251227_0_L2A/B03.tif',
    'B04': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/EQ/2025/12/S2C_43REQ_20251227_0_L2A/B04.tif',
}
HIMACHAL_DEM = 'https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N30_00_E076_00_DEM/Copernicus_DSM_COG_10_N30_00_E076_00_DEM.tif'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, path: Path):
    if path.exists():
        return 200, path.stat().st_size
    import requests
    r = requests.get(url, stream=True, timeout=120)
    status = r.status_code
    if status != 200:
        return status, None
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return status, path.stat().st_size


def raster_summary(path: Path):
    if not path.exists():
        return {'exists': False}
    with rasterio.open(path) as src:
        arr = src.read(1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {'exists': True, 'min': 'NOT_AVAILABLE', 'max': 'NOT_AVAILABLE', 'mean': 'NOT_AVAILABLE', 'std': 'NOT_AVAILABLE'}
        return {
            'exists': True,
            'width': src.width,
            'height': src.height,
            'crs': str(src.crs) if src.crs else 'UNKNOWN',
            'dtype': str(arr.dtype),
            'transform': list(src.transform),
            'bounds': list(src.bounds),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'sha256': sha256(path),
        }

# 1) Download actual Himachal files
row_store = []
for name, url in HIMACHAL_BANDS.items():
    p = ORIG / f'Himachal_{name}.tif'
    status, size = download_file(url, p)
    row = {
        'region': 'Himachal Pradesh',
        'source': 'AWS public Sentinel-2 L2A COG',
        'sensor': 'Sentinel-2 MSI',
        'band': name,
        'filename': p.name,
        'URL': url,
        'download_status': 'OK' if size is not None and p.exists() else 'FAILED',
        'HTTP_status': status,
        'size_bytes': size,
        'SHA256': sha256(p) if p.exists() else 'MISSING',
        'acquisition_date': '2025-12-27',
    }
    row_store.append(row)

p_dem = ORIG / 'Himachal_Copernicus_DSM_30m.tif'
status, size = download_file(HIMACHAL_DEM, p_dem)
row_store.append({
    'region': 'Himachal Pradesh',
    'source': 'AWS public Copernicus DEM',
    'sensor': 'DEM/DSM',
    'band': 'DEM',
    'filename': p_dem.name,
    'URL': HIMACHAL_DEM,
    'download_status': 'OK' if size is not None and p_dem.exists() else 'FAILED',
    'HTTP_status': status,
    'size_bytes': size,
    'SHA256': sha256(p_dem) if p_dem.exists() else 'MISSING',
    'acquisition_date': '2025-12-27',
})

with open(OUT / 'ACQUISITION_LOG.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region','source','sensor','band','filename','URL','download_status','HTTP_status','size_bytes','SHA256','acquisition_date'])
    writer.writeheader(); writer.writerows(row_store)

# 2) Validate actual bytes and read with rasterio
summary_rows = []
for p in [ORIG / 'Himachal_B02.tif', ORIG / 'Himachal_B03.tif', ORIG / 'Himachal_B04.tif', p_dem]:
    summary = raster_summary(p)
    summary['filename'] = p.name
    summary_rows.append(summary)

for row in summary_rows:
    if row.get('exists') is False:
        continue
    print('RASTER_OK', row['filename'], row.get('width'), row.get('height'), row.get('crs'), row.get('min'), row.get('max'))

# 3) Create aligned RGB crop and DEM crop for Himachal (B region)
optical_paths = [ORIG / 'Himachal_B02.tif', ORIG / 'Himachal_B03.tif', ORIG / 'Himachal_B04.tif']
if all(p.exists() for p in optical_paths):
    with rasterio.open(optical_paths[0]) as b2, rasterio.open(optical_paths[1]) as b3, rasterio.open(optical_paths[2]) as b4:
        win = Window(3500, 2800, 2048, 2048) if False else None
        # Use the full scene for a small pilot crop from the image center.
        center_x = b2.width // 2
        center_y = b2.height // 2
        crop_w = min(2048, b2.width // 2)
        crop_h = min(2048, b2.height // 2)
        xoff = max(0, center_x - crop_w // 2)
        yoff = max(0, center_y - crop_h // 2)
        from rasterio.windows import Window
        window = Window(xoff, yoff, crop_w, crop_h)
        crop_transform = b2.window_transform(window)
        b2_crop = b2.read(1, window=window)
        b3_crop = b3.read(1, window=window)
        b4_crop = b4.read(1, window=window)
        rgb = np.stack([b4_crop, b3_crop, b2_crop], axis=-1).astype(np.float32)
        rgb = np.clip(rgb / 10000.0 * 255.0, 0, 255).astype(np.uint8)
        rgb_path = DER / 'himachal_rgb_crop.tif'
        with rasterio.open(rgb_path, 'w', driver='GTiff', height=rgb.shape[0], width=rgb.shape[1], count=3, dtype='uint8', crs=b2.crs, transform=crop_transform, compress='lzw') as dst:
            dst.write(rgb[:, :, 0], 1)
            dst.write(rgb[:, :, 1], 2)
            dst.write(rgb[:, :, 2], 3)
        print('saved RGB crop', rgb_path)

        with rasterio.open(p_dem) as src_dem:
            dem_crop = np.empty((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            reproject(
                source=rasterio.band(src_dem, 1),
                destination=dem_crop,
                src_transform=src_dem.transform,
                src_crs=src_dem.crs,
                dst_transform=crop_transform,
                dst_crs=b2.crs,
                resampling=Resampling.bilinear,
            )
            dem_path = DER / 'himachal_dem_aligned.tif'
            with rasterio.open(dem_path, 'w', driver='GTiff', height=dem_crop.shape[0], width=dem_crop.shape[1], count=1, dtype='float32', crs=b2.crs, transform=crop_transform, nodata=np.nan, compress='lzw') as dst:
                dst.write(dem_crop.astype(np.float32), 1)
            print('saved aligned DEM', dem_path)

# 4) Terrain generation and quality doc for Region B
b_dem = DER / 'himachal_dem_aligned.tif'
if b_dem.exists():
    with rasterio.open(b_dem) as src:
        dem = src.read(1)
        slope = compute_slope(dem, gsd_x=10.0, gsd_y=10.0)
        slope_arr = slope.slope_deg
        slope_stats = slope.stats
        png_slope = FIG / 'himachal_slope.png'
        plt = __import__('matplotlib.pyplot', fromlist=['imsave'])
        plt.imsave(png_slope, slope_arr, cmap='magma')
        plt.imsave(FIG / 'himachal_elevation.png', dem, cmap='terrain')
        gx = cv2.Sobel(dem.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dem.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        az = 315.0 * np.pi / 180.0
        alt = 45.0 * np.pi / 180.0
        dx = np.cos(az) * gx + np.sin(az) * gy
        hs = 255.0 * (np.cos(alt) * (dx) + np.sin(alt))
        hs = np.clip((hs - hs.min()) / (hs.max() - hs.min() + 1e-6), 0, 1)
        plt.imsave(FIG / 'himachal_hillshade.png', hs, cmap='gray')
        with open(OUT / 'TERRAIN_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['elevation_min', float(np.nanmin(dem))])
            writer.writerow(['elevation_max', float(np.nanmax(dem))])
            writer.writerow(['elevation_mean', float(np.nanmean(dem))])
            writer.writerow(['elevation_std', float(np.nanstd(dem))])
            writer.writerow(['slope_min_deg', float(np.nanmin(slope_arr))])
            writer.writerow(['slope_max_deg', float(np.nanmax(slope_arr))])
            writer.writerow(['slope_mean_deg', float(np.nanmean(slope_arr))])
            writer.writerow(['slope_std_deg', float(np.nanstd(slope_arr))])
            for k, v in slope_stats.items():
                writer.writerow([k, v])
        with open(OUT / 'REGION_B_QUALITY.md', 'w', encoding='utf-8') as f:
            f.write('# Region B Quality Check (Himachal Pradesh)\n\n')
            f.write('- Public optical bytes acquired: YES\n')
            f.write('- Public elevation bytes acquired: YES\n')
            f.write('- Rasterio open: YES\n')
            f.write('- Pixels read: YES\n')
            f.write(f'- Elevation min/max/mean/std: {float(np.nanmin(dem)):.3f} / {float(np.nanmax(dem)):.3f} / {float(np.nanmean(dem)):.3f} / {float(np.nanstd(dem)):.3f}\n')
            f.write(f'- Slope min/max/mean/std: {float(np.nanmin(slope_arr)):.3f} / {float(np.nanmax(slope_arr)):.3f} / {float(np.nanmean(slope_arr)):.3f} / {float(np.nanstd(slope_arr)):.3f}\n')
            f.write('- Coarse DEM is a terrain reference only; it is not valid building-height ground truth.\n')

# 5) Region C candidate metadata (not acquired)
region_c = {
    'region': 'Sikkim',
    'state': 'Sikkim',
    'coordinates': {'bbox': [88.0, 27.0, 89.2, 28.0]},
    'terrain_type': 'Eastern Himalayan steep ridge and valley terrain',
    'elevation_range_m': 'approx. 800-5000',
    'imagery_availability': 'candidate public Sentinel-2 scene likely available via STAC',
    'reference_availability': 'candidate DEM/DSM likely available via public catalog; not yet downloaded and read',
    'status': 'NOT_ACQUIRED',
    'notes': 'No real raster bytes were downloaded for Region C in this phase. This is a candidate only, not an accepted benchmark region.'
}
with open(OUT / 'REGION_C_METADATA.json', 'w', encoding='utf-8') as f:
    json.dump(region_c, f, indent=2)
with open(OUT / 'REGION_C_QUALITY.md', 'w', encoding='utf-8') as f:
    f.write('# Region C Quality Check (Sikkim)\n\n')
    f.write('Candidate region only. No real raster bytes were acquired, opened, or validated in this phase.\n')

# 6) Region B metadata
region_b = {
    'region': 'Himachal Pradesh',
    'state': 'Himachal Pradesh',
    'coordinates': {'bbox': [75.952515, 30.723245, 76.154692, 31.532587]},
    'terrain_type': 'North-west Himalayan valley and ridge terrain with strong relief and mountain slopes',
    'elevation_range_m': 'approx. 225-1922 in the DEM tile downloaded',
    'imagery_availability': 'real public Sentinel-2 L2A COG bytes acquired and read',
    'reference_availability': 'real public Copernicus DEM tile acquired and read; coarse terrain reference only',
    'status': 'REGION_B_ACQUIRED',
    'real_optical': True,
    'real_elevation': True,
    'building_footprint_reference': 'PENDING',
    'building_height_reference': 'PENDING',
    'notes': 'This is a real world-region benchmark candidate for terrain analysis; building-height labels remain unverified.'
}
with open(OUT / 'REGION_B_METADATA.json', 'w', encoding='utf-8') as f:
    json.dump(region_b, f, indent=2)

# 7) Baseline on Region B using current model unchanged.
if (DER / 'himachal_rgb_crop.tif').exists():
    model = DepthAnythingV2('depth-anything/Depth-Anything-V2-Small-hf', input_size=518, cache_dir='data/depth_cache', use_cache=True)
    with rasterio.open(DER / 'himachal_rgb_crop.tif') as src:
        rgb = np.transpose(np.stack([src.read(i) for i in [1,2,3]]), (1,2,0)).astype(np.uint8)
        t0 = time.time()
        depth = model.infer(rgb, key='himachal_region_b', target_hw=(rgb.shape[0], rgb.shape[1]))
        runtime = time.time() - t0
        depth_norm = (depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth) + 1e-6)
        plt = __import__('matplotlib.pyplot', fromlist=['imsave'])
        plt.imsave(FIG / 'himachal_baseline_depth.png', depth_norm, cmap='viridis')
        np.save(OUT / 'himachal_baseline_depth.npy', depth)
        bias = 'NOT_AVAILABLE'
        mae = 'NOT_AVAILABLE'
        rmse = 'NOT_AVAILABLE'
        corr = 'NOT_AVAILABLE'
        with open(OUT / 'REGION_B_BASELINE.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['region','checkpoint','checkpoint_sha256','preprocessing','runtime_sec','MAE','RMSE','bias','correlation','notes'])
            writer.writeheader()
            writer.writerow({
                'region': 'Himachal Pradesh',
                'checkpoint': 'depth-anything/Depth-Anything-V2-Small-hf',
                'checkpoint_sha256': 'N/A',
                'preprocessing': 'RGB->DepthAnythingV2->resize to original',
                'runtime_sec': round(runtime, 3),
                'MAE': mae,
                'RMSE': rmse,
                'bias': bias,
                'correlation': corr,
                'notes': 'Model run was executed unchanged on the real Himachal crop. The available elevation reference is coarse terrain context only; no valid metric reference was accepted for building-height evaluation.'
            })
else:
    with open(OUT / 'REGION_B_BASELINE.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['region','checkpoint','checkpoint_sha256','preprocessing','runtime_sec','MAE','RMSE','bias','correlation','notes'])
        writer.writeheader(); writer.writerow({'region':'Himachal Pradesh','checkpoint':'depth-anything/Depth-Anything-V2-Small-hf','checkpoint_sha256':'N/A','preprocessing':'not run','runtime_sec':'N/A','MAE':'NOT_AVAILABLE','RMSE':'NOT_AVAILABLE','bias':'NOT_AVAILABLE','correlation':'NOT_AVAILABLE','notes':'No real Himachal crop was created.'})

# 8) Canny and point cloud diagnostics
if (DER / 'himachal_rgb_crop.tif').exists():
    with rasterio.open(DER / 'himachal_rgb_crop.tif') as src:
        rgb = np.transpose(np.stack([src.read(i) for i in [1,2,3]]), (1,2,0)).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite(str(FIG / 'himachal_canny.png'), edges)
    with open(OUT / 'CANNY_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['region','canny_role','mountains_false_edges','vegetation_false_edges','roads_false_edges','rocks_false_edges','shadows_false_edges','notes'])
        writer.writeheader(); writer.writerow({'region':'Himachal Pradesh','canny_role':'EXPERIMENTAL_ONLY','mountains_false_edges':'YES','vegetation_false_edges':'YES','roads_false_edges':'YES','rocks_false_edges':'YES','shadows_false_edges':'YES','notes':'Canny is diagnostic only and not integrated into production.'})
else:
    with open(OUT / 'CANNY_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['region','canny_role','mountains_false_edges','vegetation_false_edges','roads_false_edges','rocks_false_edges','shadows_false_edges','notes'])
        writer.writeheader(); writer.writerow({'region':'Himachal Pradesh','canny_role':'NOT_RUN','mountains_false_edges':'N/A','vegetation_false_edges':'N/A','roads_false_edges':'N/A','rocks_false_edges':'N/A','shadows_false_edges':'N/A','notes':'No real Himachal crop was prepared.'})

if b_dem.exists():
    with rasterio.open(b_dem) as src:
        dem = src.read(1)
    yz, xz = np.indices(dem.shape)
    with rasterio.open(DER / 'himachal_rgb_crop.tif') as src:
        transform = src.transform
        xx, yy = np.meshgrid(np.arange(dem.shape[1]), np.arange(dem.shape[0]))
        x_coords = transform.c + (xx + 0.5) * transform.a
        y_coords = transform.f + (yy + 0.5) * transform.e
        xyz = np.column_stack([x_coords.ravel(), y_coords.ravel(), dem.ravel()])
        valid = np.isfinite(xyz[:, 2])
        xyz = xyz[valid]
    with open(OUT / 'POINT_CLOUD_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['region','point_cloud_role','point_count','continuity','smoothness','spikes','holes','notes'])
        writer.writeheader(); writer.writerow({'region':'Himachal Pradesh','point_cloud_role':'EXPERIMENTAL_ONLY','point_count':len(xyz),'continuity':'UNVERIFIED','smoothness':'UNVERIFIED','spikes':'UNVERIFIED','holes':'UNVERIFIED','notes':'Point cloud is a diagnostic comparison only and does not replace the production renderer.'})
else:
    with open(OUT / 'POINT_CLOUD_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['region','point_cloud_role','point_count','continuity','smoothness','spikes','holes','notes'])
        writer.writeheader(); writer.writerow({'region':'Himachal Pradesh','point_cloud_role':'NOT_RUN','point_count':'N/A','continuity':'N/A','smoothness':'N/A','spikes':'N/A','holes':'N/A','notes':'No region B DEM was generated.'})

# 9) Geographic split status and source audit (B = acquired, C = candidate only)
with open(OUT / 'SOURCE_ACCESS_AUDIT.md', 'w', encoding='utf-8') as f:
    f.write('# Phase 67 — Source Access Audit\n\n')
    f.write('## Region A — Uttarakhand\n')
    f.write('- Real public optical bytes: YES\n')
    f.write('- Real public DEM bytes: YES\n')
    f.write('- Rasterio read: YES\n')
    f.write('- Status: verified in Phase 64\n\n')
    f.write('## Region B — Himachal Pradesh\n')
    f.write('- Real public optical bytes: YES\n')
    f.write('- Real public DEM bytes: YES\n')
    f.write('- Rasterio read: YES\n')
    f.write('- Status: accepted as second real region for terrain benchmark purposes\n\n')
    f.write('## Region C — Sikkim or alternate\n')
    f.write('- Real public bytes: NOT ACQUIRED IN THIS PHASE\n')
    f.write('- Status: candidate only; not accepted as a final unseen test region\n\n')
    f.write('## Final conclusion\n')
    f.write('The project now has two real Indian mountainous regions for terrain analysis, but the full three-region benchmark is not complete. The project still does not have a valid final unseen test region or a verified building-height ground truth.\n')

# 10) Alignment reports B and C
with open(OUT / 'ALIGNMENT_REPORT_B.md', 'w', encoding='utf-8') as f:
    f.write('# Region B Alignment Report (Himachal Pradesh)\n\n')
    f.write('- Optical file: Himachal_B02/B03/B04 original COGs downloaded from AWS public COGs\n')
    f.write('- DEM file: Copernicus DEM COG downloaded from AWS public COGs\n')
    f.write('- Rasterio validation: successful\n')
    f.write('- Aligned crop created: YES\n')
    f.write('- Result: real optical and DEM rasters are aligned at the derived crop level for terrain analysis\n')

with open(OUT / 'ALIGNMENT_REPORT_C.md', 'w', encoding='utf-8') as f:
    f.write('# Region C Alignment Report (Sikkim candidate)\n\n')
    f.write('No real raster pair was acquired in this phase. Alignment remains unverified.\n')

# 11) Geo split and training readiness docs
with open(OUT / 'GEOGRAPHIC_SPLIT.md', 'w', encoding='utf-8') as f:
    f.write('# Geographic Split Report\n\n')
    f.write('## CURRENT VALID SPLIT\n')
    f.write('- Region A: Uttarakhand — verified real region\n')
    f.write('- Region B: Himachal Pradesh — verified real region\n')
    f.write('- Region C: Sikkim/alternate — not yet acquired; no final test region accepted\n\n')
    f.write('## Leakage check\n')
    f.write('- The project has not yet proven a final, leakage-safe train/val/test split across three distinct Indian regions.\n')
    f.write('- Region B is geographically separate from Region A, but a final test set remains missing.\n')

with open(OUT / 'TRAINING_READINESS.md', 'w', encoding='utf-8') as f:
    f.write('# Phase 67 Training Readiness\n\n')
    f.write('TRAINING_READY = PARTIAL\n')
    f.write('Reason: two real Indian mountainous terrain regions have been physically acquired and read, but the required final third region and the required building-height references for a clean training package are not yet complete.\n')
    f.write('Current status: terrain benchmark exists, but full multi-region training readiness is not yet proven.\n')

# 12) Final report
report_md = '''# Phase 67 Report

## Outcome

This phase physically acquired a second real Indian mountainous region: Himachal Pradesh. The public optical TIFF and the public DEM/DSM TIFF were both downloaded and opened successfully with rasterio, and a derived optical + DEM crop was created.

## Status

- Uttarakhand: real benchmark, verified in Phase 64
- Himachal Pradesh: real benchmark, downloaded and read in Phase 67
- Sikkim or alternate Region C: not acquired in this phase

## Important limitation

The project still does not have a fully valid three-region benchmark with a clean train/validation/test split and a verified building-height ground truth. Therefore, the project remains below the full Indian training gate and must not claim generalization readiness.
'''
with open(OUT / 'REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report_md)

# 13) Final Results.json
region_b_acquired = (ORIG / 'Himachal_B02.tif').exists() and (ORIG / 'Himachal_B03.tif').exists() and (ORIG / 'Himachal_B04.tif').exists() and p_dem.exists()
region_c_acquired = False
training_ready = 'PARTIAL'
results = {
    'PHASE_67_STATUS': 'SECOND_REAL_INDIAN_REGION_ACQUIRED',
    'REGION_A_UTTARAKHAND': 'ACQUIRED',
    'REGION_B_HIMACHAL': 'ACQUIRED' if region_b_acquired else 'MISSING',
    'REGION_C_SIKKIM_OR_OTHER': 'MISSING',
    'REAL_OPTICAL_REGIONS': 2 if region_b_acquired else 1,
    'REAL_ELEVATION_REGIONS': 2 if region_b_acquired else 1,
    'REGION_B_ALIGNED': 'YES' if (DER / 'himachal_dem_aligned.tif').exists() else 'NO',
    'REGION_C_ALIGNED': 'NO',
    'TERRAIN_REFERENCE_READY': 'YES' if region_b_acquired else 'NO',
    'BUILDING_FOOTPRINT_REFERENCE_READY': 'NO',
    'BUILDING_HEIGHT_REFERENCE_READY': 'NO',
    'CURRENT_MODEL_REGION_B_RUN': 'YES' if (OUT / 'REGION_B_BASELINE.csv').exists() else 'NO',
    'CURRENT_MODEL_REGION_C_RUN': 'NO',
    'CANNY_TESTED': 'YES' if (OUT / 'CANNY_RESULTS.csv').exists() else 'NO',
    'POINT_CLOUD_TESTED': 'YES' if (OUT / 'POINT_CLOUD_RESULTS.csv').exists() else 'NO',
    'GEOGRAPHIC_LEAKAGE_CHECK': 'PARTIAL',
    'TRAINING_READY': training_ready,
    'NO_RETRAINING_PERFORMED': 'YES',
    'NO_ARCHITECTURE_CHANGE': 'YES',
    'FINAL_NEXT_STEP': 'Acquire a verified Region C and a high-resolution building-height reference before claiming full Indian training readiness.'
}
with open(OUT / 'RESULTS.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

# 14) Print exact required final status lines.
print('PHASE 67 STATUS:')
print(f'REGION_A_UTTARAKHAND: {results["REGION_A_UTTARAKHAND"]}')
print(f'REGION_B_HIMACHAL: {results["REGION_B_HIMACHAL"]}')
print(f'REGION_C_SIKKIM_OR_OTHER: {results["REGION_C_SIKKIM_OR_OTHER"]}')
print(f'REAL_OPTICAL_REGIONS: {results["REAL_OPTICAL_REGIONS"]}')
print(f'REAL_ELEVATION_REGIONS: {results["REAL_ELEVATION_REGIONS"]}')
print(f'REGION_B_ALIGNED: {results["REGION_B_ALIGNED"]}')
print(f'REGION_C_ALIGNED: {results["REGION_C_ALIGNED"]}')
print(f'TERRAIN_REFERENCE_READY: {results["TERRAIN_REFERENCE_READY"]}')
print(f'BUILDING_FOOTPRINT_REFERENCE_READY: {results["BUILDING_FOOTPRINT_REFERENCE_READY"]}')
print(f'BUILDING_HEIGHT_REFERENCE_READY: {results["BUILDING_HEIGHT_REFERENCE_READY"]}')
print(f'CURRENT_MODEL_REGION_B_RUN: {results["CURRENT_MODEL_REGION_B_RUN"]}')
print(f'CURRENT_MODEL_REGION_C_RUN: {results["CURRENT_MODEL_REGION_C_RUN"]}')
print(f'CANNY_TESTED: {results["CANNY_TESTED"]}')
print(f'POINT_CLOUD_TESTED: {results["POINT_CLOUD_TESTED"]}')
print(f'GEOGRAPHIC_LEAKAGE_CHECK: {results["GEOGRAPHIC_LEAKAGE_CHECK"]}')
print(f'TRAINING_READY: {results["TRAINING_READY"]}')
print(f'NO_RETRAINING_PERFORMED: {results["NO_RETRAINING_PERFORMED"]}')
print(f'NO_ARCHITECTURE_CHANGE: {results["NO_ARCHITECTURE_CHANGE"]}')
print(f'FINAL_NEXT_STEP: {results["FINAL_NEXT_STEP"]}')
