from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import rasterio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PH64 = ROOT / 'runs' / 'phase64_indian_data'
PH65 = ROOT / 'runs' / 'phase65_indian_multiregion'
OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

# Preserve historical evidence by copying existing Phase 64 visual material into this package.
for name in ['baseline_depth.png', 'canny_edges.png', 'elevation.png', 'hillshade.png', 'slope.png']:
    src = PH64 / 'figures' / name
    if src.exists():
        shutil.copy2(src, FIG / name)

# Phase 64 verified evidence
phase64_results = {}
if (PH64 / 'RESULTS.json').exists():
    phase64_results = json.loads((PH64 / 'RESULTS.json').read_text(encoding='utf-8'))
phase65_results = {}
if (PH65 / 'RESULTS.json').exists():
    phase65_results = json.loads((PH65 / 'RESULTS.json').read_text(encoding='utf-8'))

# Helper for numeric stats

def raster_stats(path: Path):
    if not path.exists():
        return {'exists': False, 'min': 'NOT_AVAILABLE', 'max': 'NOT_AVAILABLE', 'mean': 'NOT_AVAILABLE', 'std': 'NOT_AVAILABLE'}
    with rasterio.open(path) as src:
        arr = src.read(1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {'exists': True, 'min': 'NOT_AVAILABLE', 'max': 'NOT_AVAILABLE', 'mean': 'NOT_AVAILABLE', 'std': 'NOT_AVAILABLE'}
        return {
            'exists': True,
            'min': float(arr.min()),
            'max': float(arr.max()),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
        }

# 1) Source Access Audit
source_audit = '''# Phase 66 — Source Access Audit

## Verified historical evidence

### Region A: Uttarakhand (verified)
- Optical source: Sentinel-2 L2A AWS public COG
- Elevation source: Copernicus DEM GLO-30 AWS public COG
- Verification status: downloaded, opened with rasterio, pixels read successfully
- Evidence: Phase 64 downloaded actual bytes and recorded SHA256 checksums
- Limitation: this DEM is too coarse for high-resolution building-height validation; it is acceptable as terrain context only

### Region B: Himachal Pradesh (candidate, not yet accepted)
- Optical source: publicly indexed Sentinel-2 scenes are expected to exist through Copernicus / STAC access
- Elevation source: public DEM collections and regional products may exist, but no real paired raster was downloaded and read in this workspace
- Verification status: not accepted as a benchmark yet
- Limitation: no actual files, no pixel-read validation, no alignment proof

### Region C: Sikkim (candidate, not yet accepted)
- Optical source: publicly indexed Sentinel-2 scenes are expected to exist through Copernicus / STAC access
- Elevation source: public DEM collections may exist, but no real paired raster was downloaded and read in this workspace
- Verification status: not accepted as a benchmark yet
- Limitation: no actual files, no pixel-read validation, no alignment proof

## Ground-truth hierarchy used

- Level 1: LiDAR / high-resolution photogrammetric DSM — not currently available for the selected Indian benchmark regions in this workspace
- Level 2: high-resolution DSM / DTM — not yet acquired for Himachal or Sikkim
- Level 3: medium-resolution DEM — yes, Uttarakhand has a public Copernicus DEM reference, but it is a terrain reference only
- Level 4: coarse DEM — used only for terrain context, not for building-height ground truth

## Building reference status

- Building footprints: not verified in any accepted Indian region
- Building height labels: not verified in any accepted Indian region
- DSM-derived building heights: not verified in any accepted Indian region

## Overall conclusion

The project has one real terrain benchmark and no accepted high-resolution building ground truth. The benchmark is therefore not yet training-ready for building-specific work and must remain on the evidence gate until the reference data is physically acquired and aligned.
'''
with open(OUT / 'SOURCE_ACCESS_AUDIT.md', 'w', encoding='utf-8') as f:
    f.write(source_audit)

# 2) Dataset matrix
matrix_rows = [
    {
        'region': 'Uttarakhand',
        'state': 'Uttarakhand',
        'optical_source': 'Sentinel-2 L2A AWS Public COG',
        'optical_resolution': '10 m',
        'elevation_source': 'Copernicus DEM GLO-30 AWS Public COG',
        'elevation_resolution': '30 m',
        'elevation_type': 'DEM',
        'building_labels': 'NO',
        'building_height_labels': 'NO',
        'slope_available': 'YES',
        'hazard_labels': 'NO',
        'CRS': 'EPSG:32644 (optical); EPSG:4326 (source DEM)',
        'date': '2026-08-30',
        'license': 'Open public access',
        'access': 'downloaded and rasterio-read verified',
        'training_suitability': 'TERRAIN_ONLY',
        'validation_suitability': 'TERRAIN_ONLY',
        'test_suitability': 'TERRAIN_ONLY',
        'limitations': 'No accepted high-res building reference; DEM is coarse and must not be mislabeled as building-height ground truth.'
    },
    {
        'region': 'Himachal Pradesh',
        'state': 'Himachal Pradesh',
        'optical_source': 'Sentinel-2 public catalog candidate',
        'optical_resolution': '10 m',
        'elevation_source': 'Public DEM / regional candidate',
        'elevation_resolution': 'Pending verification',
        'elevation_type': 'Candidate DEM/DSM',
        'building_labels': 'PENDING',
        'building_height_labels': 'PENDING',
        'slope_available': 'PENDING',
        'hazard_labels': 'PENDING',
        'CRS': 'Scene dependent',
        'date': 'Pending verification',
        'license': 'Public / dataset dependent',
        'access': 'not yet downloaded and read',
        'training_suitability': 'PENDING',
        'validation_suitability': 'PENDING',
        'test_suitability': 'PENDING',
        'limitations': 'No physical raster pair has yet been acquired and aligned in this workspace.'
    },
    {
        'region': 'Sikkim',
        'state': 'Sikkim',
        'optical_source': 'Sentinel-2 public catalog candidate',
        'optical_resolution': '10 m',
        'elevation_source': 'Public DEM / regional candidate',
        'elevation_resolution': 'Pending verification',
        'elevation_type': 'Candidate DEM/DSM',
        'building_labels': 'PENDING',
        'building_height_labels': 'PENDING',
        'slope_available': 'PENDING',
        'hazard_labels': 'PENDING',
        'CRS': 'Scene dependent',
        'date': 'Pending verification',
        'license': 'Public / dataset dependent',
        'access': 'not yet downloaded and read',
        'training_suitability': 'PENDING',
        'validation_suitability': 'PENDING',
        'test_suitability': 'PENDING',
        'limitations': 'No physical raster pair has yet been acquired and aligned in this workspace.'
    },
]
with open(OUT / 'INDIAN_DATASET_MATRIX.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region','state','optical_source','optical_resolution','elevation_source','elevation_resolution','elevation_type','building_labels','building_height_labels','slope_available','hazard_labels','CRS','date','license','access','training_suitability','validation_suitability','test_suitability','limitations'])
    writer.writeheader(); writer.writerows(matrix_rows)

# 3) Terrain reference
utm_dem = PH64 / 'DERIVED_DATA' / 'uttarakhand_dem_aligned.tif'
terrain_rows = [
    {
        'region': 'Uttarakhand',
        'terrain_source': 'Copernicus DEM GLO-30 (reprojected crop)',
        'elevation_read_verified': 'YES',
        'dem_min': raster_stats(utm_dem)['min'],
        'dem_max': raster_stats(utm_dem)['max'],
        'dem_mean': raster_stats(utm_dem)['mean'],
        'dem_std': raster_stats(utm_dem)['std'],
        'slope_min_deg': 'NOT_AVAILABLE',
        'slope_max_deg': 'NOT_AVAILABLE',
        'slope_mean_deg': 'NOT_AVAILABLE',
        'slope_std_deg': 'NOT_AVAILABLE',
        'notes': 'Raster exists and was read, but the DEM is too coarse for building-height ground truth.'
    },
    {
        'region': 'Himachal Pradesh',
        'terrain_source': 'CANDIDATE_PUBLIC_DEM',
        'elevation_read_verified': 'NO',
        'dem_min': 'NOT_AVAILABLE',
        'dem_max': 'NOT_AVAILABLE',
        'dem_mean': 'NOT_AVAILABLE',
        'dem_std': 'NOT_AVAILABLE',
        'slope_min_deg': 'NOT_AVAILABLE',
        'slope_max_deg': 'NOT_AVAILABLE',
        'slope_mean_deg': 'NOT_AVAILABLE',
        'slope_std_deg': 'NOT_AVAILABLE',
        'notes': 'Not yet acquired or validated.'
    },
    {
        'region': 'Sikkim',
        'terrain_source': 'CANDIDATE_PUBLIC_DEM',
        'elevation_read_verified': 'NO',
        'dem_min': 'NOT_AVAILABLE',
        'dem_max': 'NOT_AVAILABLE',
        'dem_mean': 'NOT_AVAILABLE',
        'dem_std': 'NOT_AVAILABLE',
        'slope_min_deg': 'NOT_AVAILABLE',
        'slope_max_deg': 'NOT_AVAILABLE',
        'slope_mean_deg': 'NOT_AVAILABLE',
        'slope_std_deg': 'NOT_AVAILABLE',
        'notes': 'Not yet acquired or validated.'
    },
]
with open(OUT / 'TERRAIN_REFERENCE.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region','terrain_source','elevation_read_verified','dem_min','dem_max','dem_mean','dem_std','slope_min_deg','slope_max_deg','slope_mean_deg','slope_std_deg','notes'])
    writer.writeheader(); writer.writerows(terrain_rows)

# 4) Building reference
building_rows = [
    {'region': 'Uttarakhand', 'building_footprint_reference': 'NO', 'building_height_reference': 'NO', 'source': 'None accepted', 'level': 'NONE', 'notes': 'No high-resolution building footprint or height labels available in the verified benchmark.'},
    {'region': 'Himachal Pradesh', 'building_footprint_reference': 'PENDING', 'building_height_reference': 'PENDING', 'source': 'Candidate public datasets', 'level': 'PENDING', 'notes': 'Not yet downloaded and aligned.'},
    {'region': 'Sikkim', 'building_footprint_reference': 'PENDING', 'building_height_reference': 'PENDING', 'source': 'Candidate public datasets', 'level': 'PENDING', 'notes': 'Not yet downloaded and aligned.'},
]
with open(OUT / 'BUILDING_REFERENCE.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region','building_footprint_reference','building_height_reference','source','level','notes'])
    writer.writeheader(); writer.writerows(building_rows)

# 5) Train/val/test manifest
manifest_rows = [
    {'split': 'TRAIN', 'region': 'Uttarakhand', 'state': 'Uttarakhand', 'optical_status': 'REAL', 'elevation_status': 'REAL_COARSE_DEM', 'building_gt_status': 'NO', 'leakage_check': 'PASS (only one region exists)', 'notes': 'Real benchmark pair exists; no building-height ground truth.'},
    {'split': 'VALIDATION', 'region': 'Himachal Pradesh', 'state': 'Himachal Pradesh', 'optical_status': 'CANDIDATE', 'elevation_status': 'CANDIDATE', 'building_gt_status': 'PENDING', 'leakage_check': 'UNVERIFIED', 'notes': 'Geographically separate candidate only; physical acquisition pending.'},
    {'split': 'TEST', 'region': 'Sikkim', 'state': 'Sikkim', 'optical_status': 'CANDIDATE', 'elevation_status': 'CANDIDATE', 'building_gt_status': 'PENDING', 'leakage_check': 'UNVERIFIED', 'notes': 'Geographically separate candidate only; physical acquisition pending.'},
]
with open(OUT / 'TRAIN_VALIDATION_TEST_MANIFEST.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['split','region','state','optical_status','elevation_status','building_gt_status','leakage_check','notes'])
    writer.writeheader(); writer.writerows(manifest_rows)

# 6) Data alignment quality report
alignment_report = '''# Phase 66 — Data Alignment Report

## Verified alignment

- Region A (Uttarakhand): YES — real public optical and DEM rasters were downloaded, aligned, and a derived crop was created in the Phase 64 pipeline.
- Region B (Himachal Pradesh): NO — no real paired raster pair was downloaded or aligned in this workspace.
- Region C (Sikkim): NO — no real paired raster pair was downloaded or aligned in this workspace.

## Summary

A valid geospatial alignment pipeline exists for the verified Uttarakhand pilot, but the full three-region benchmark has not yet been physically assembled. The benchmark remains incomplete, and training readiness is not supported by the evidence.
'''
with open(OUT / 'DATA_ALIGNMENT_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(alignment_report)

# 7) Data quality report
quality_report = '''# Phase 66 — Data Quality Report

## Terrain quality

- Uttarakhand: real optical file exists; real DEM exists; pixels were read; alignment is verified.
- Uttarakhand DEM source is coarse (30m) and suitable only for terrain context / slope context, not building-height evaluation.
- No high-resolution height labels or L1/L2 building labels were identified.

## Building quality

- Building footprint labels: not verified.
- Building height labels: not verified.
- A building benchmark cannot be claimed without physically acquired footprint and height references.

## Final conclusion

The evidence supports a terrain benchmark only, not a building benchmark or training-ready Indian dataset.
'''
with open(OUT / 'DATA_QUALITY_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(quality_report)

# 8) India baseline results from verified Phase 64 run; copied unchanged.
phase64_baseline = []
if (PH64 / 'BASELINE_RESULTS.csv').exists():
    with open(PH64 / 'BASELINE_RESULTS.csv', newline='', encoding='utf-8') as f:
        phase64_baseline = list(csv.DictReader(f))
if not phase64_baseline:
    phase64_baseline = [{
        'dataset': 'Uttarakhand real crop',
        'task': 'relative depth inference',
        'checkpoint': 'depth-anything/Depth-Anything-V2-Small-hf',
        'checkpoint_sha256': 'N/A',
        'preprocessing': 'RGB->DepthAnythingV2->resize to original',
        'input_dimensions': '2048x2048',
        'output_dimensions': '2048x2048',
        'runtime_sec': 'real run in Phase 64',
        'MAE': 'NOT_AVAILABLE',
        'RMSE': 'NOT_AVAILABLE',
        'bias': 'NOT_AVAILABLE',
        'correlation': 'NOT_AVAILABLE',
        'notes': 'No accepted terrain or building metric reference was available.'
    }]
with open(OUT / 'INDIA_BASELINE_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(phase64_baseline[0].keys()))
    writer.writeheader(); writer.writerows(phase64_baseline)

# 9) Terrain stratified results
terrain_stratified = [
    {'region': 'Uttarakhand', 'slope_band_deg': '0-5', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'No accepted terrain ground-truth metric reference beyond coarse DEM context.'},
    {'region': 'Uttarakhand', 'slope_band_deg': '5-15', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'No accepted terrain ground-truth metric reference beyond coarse DEM context.'},
    {'region': 'Uttarakhand', 'slope_band_deg': '15-30', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'No accepted terrain ground-truth metric reference beyond coarse DEM context.'},
    {'region': 'Uttarakhand', 'slope_band_deg': '30-45', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'No accepted terrain ground-truth metric reference beyond coarse DEM context.'},
    {'region': 'Uttarakhand', 'slope_band_deg': '>45', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'No accepted terrain ground-truth metric reference beyond coarse DEM context.'},
    {'region': 'Himachal Pradesh', 'slope_band_deg': 'PENDING', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'Not yet downloaded and aligned.'},
    {'region': 'Sikkim', 'slope_band_deg': 'PENDING', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'Not yet downloaded and aligned.'},
]
with open(OUT / 'TERRAIN_STRATIFIED_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region','slope_band_deg','MAE','RMSE','bias','correlation','notes'])
    writer.writeheader(); writer.writerows(terrain_stratified)

# 10) Building results
building_results = [
    {'region': 'Uttarakhand', 'IoU': 'NOT_AVAILABLE', 'Dice': 'NOT_AVAILABLE', 'Precision': 'NOT_AVAILABLE', 'Recall': 'NOT_AVAILABLE', 'Boundary_F1': 'NOT_AVAILABLE', 'Instance_matching': 'NOT_AVAILABLE', 'Building_Height_MAE': 'NOT_AVAILABLE', 'Building_Height_RMSE': 'NOT_AVAILABLE', 'Bias': 'NOT_AVAILABLE', 'notes': 'No building reference labels available.'},
    {'region': 'Himachal Pradesh', 'IoU': 'NOT_AVAILABLE', 'Dice': 'NOT_AVAILABLE', 'Precision': 'NOT_AVAILABLE', 'Recall': 'NOT_AVAILABLE', 'Boundary_F1': 'NOT_AVAILABLE', 'Instance_matching': 'NOT_AVAILABLE', 'Building_Height_MAE': 'NOT_AVAILABLE', 'Building_Height_RMSE': 'NOT_AVAILABLE', 'Bias': 'NOT_AVAILABLE', 'notes': 'Candidate only.'},
    {'region': 'Sikkim', 'IoU': 'NOT_AVAILABLE', 'Dice': 'NOT_AVAILABLE', 'Precision': 'NOT_AVAILABLE', 'Recall': 'NOT_AVAILABLE', 'Boundary_F1': 'NOT_AVAILABLE', 'Instance_matching': 'NOT_AVAILABLE', 'Building_Height_MAE': 'NOT_AVAILABLE', 'Building_Height_RMSE': 'NOT_AVAILABLE', 'Bias': 'NOT_AVAILABLE', 'notes': 'Candidate only.'},
]
with open(OUT / 'BUILDING_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region','IoU','Dice','Precision','Recall','Boundary_F1','Instance_matching','Building_Height_MAE','Building_Height_RMSE','Bias','notes'])
    writer.writeheader(); writer.writerows(building_results)

# 11) Canny and point cloud diagnostics
canny_rows = [{
    'region': 'Uttarakhand',
    'canny_role': 'EXPERIMENTAL_ONLY',
    'footprint_boundary_help': 'UNVERIFIED',
    'roof_boundary_help': 'UNVERIFIED',
    'mountains_false_edges': 'YES',
    'vegetation_false_edges': 'YES',
    'roads_false_edges': 'YES',
    'rocks_false_edges': 'YES',
    'shadows_false_edges': 'YES',
    'notes': 'Canny is a visual diagnostic and is not integrated into production.'
}]
with open(OUT / 'CANNY_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region','canny_role','footprint_boundary_help','roof_boundary_help','mountains_false_edges','vegetation_false_edges','roads_false_edges','rocks_false_edges','shadows_false_edges','notes'])
    writer.writeheader(); writer.writerows(canny_rows)

pointcloud_rows = [{
    'region': 'Uttarakhand',
    'point_cloud_role': 'EXPERIMENTAL_ONLY',
    'terrain_continuity': 'UNVERIFIED',
    'smoothness': 'UNVERIFIED',
    'height_preservation': 'UNVERIFIED',
    'geometric_stability': 'UNVERIFIED',
    'notes': 'Point cloud is a diagnostic comparison only; it does not replace the production renderer.'
}]
with open(OUT / 'POINT_CLOUD_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region','point_cloud_role','terrain_continuity','smoothness','height_preservation','geometric_stability','notes'])
    writer.writeheader(); writer.writerows(pointcloud_rows)

# 12) Training readiness and final report
training_readiness = '''# Phase 66 — Training Readiness

## Decision

TRAINING_READY = NO
RECOMMENDED_CLASSIFICATION = INSUFFICIENT_DATA

## Evidence

- One real Indian terrain benchmark exists: Uttarakhand
- One real validation region is still missing
- One real test region is still missing
- High-resolution building-footprint ground truth is absent
- High-resolution building-height ground truth is absent
- Alignment is verified only for the Uttarakhand pilot
- Geographic leakage checks are not yet accepted for a three-region benchmark

## Recommended next phase

Acquire a second and third Indian mountainous region with real optical rasters and real elevation rasters, then verify at least one LiDAR or high-res DSM/DTM paired reference for building-height work before any training decision is made.
'''
with open(OUT / 'TRAINING_READINESS.md', 'w', encoding='utf-8') as f:
    f.write(training_readiness)

report_md = '''# Phase 66 — Indian Ground-Truth Acquisition and Benchmark Completion Report

## Summary

This phase preserved the Phase 64 and Phase 65 evidence and audited the missing benchmark ingredients required for a valid Indian training frame. The result is not a training-ready benchmark; it is a proof that the data gap is still real and must be closed before any Indian training claim is made.

## Verified evidence

- Uttarakhand optical scene exists and was read successfully.
- Uttarakhand DEM exists and was read successfully.
- Real inference was run with the current model in Phase 64.
- No accepted high-resolution building ground truth exists for the selected Indian benchmark regions.

## Scientific conclusion

The benchmark is still incomplete. The current status is BASELINE_ONLY / INSUFFICIENT_DATA, not TRAINING_READY.
'''
with open(OUT / 'REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report_md)

final_results = {
    'PHASE_66_STATUS': 'GROUND_TRUTH_GAP_REMAINS',
    'TRAIN_REGION': 'Uttarakhand',
    'VALIDATION_REGION': 'Himachal Pradesh (candidate, not yet accepted)',
    'TEST_REGION': 'Sikkim (candidate, not yet accepted)',
    'REAL_OPTICAL_REGIONS': 1,
    'REAL_ELEVATION_REGIONS': 1,
    'TERRAIN_GROUND_TRUTH': 'YES (Uttarakhand, coarse DEM / terrain context only)',
    'BUILDING_FOOTPRINT_GROUND_TRUTH': 'NO',
    'BUILDING_HEIGHT_GROUND_TRUTH': 'NO',
    'ALIGNMENT': 'A verified; B and C unverified',
    'GEOGRAPHIC_SPLIT': 'NOT_YET_ACCEPTED',
    'CURRENT_MODEL_BASELINE': 'YES_ON_REAL_UTTARAKHAND',
    'REAL_TERRAIN_METRICS': 'NO',
    'REAL_BUILDING_METRICS': 'NO',
    'CANNY_RESULT': 'EXPERIMENTAL_ONLY',
    'POINT_CLOUD_RESULT': 'EXPERIMENTAL_ONLY',
    'TRAINING_READY': 'NO',
    'RECOMMENDED_NEXT_PHASE': 'Acquire real second/third Indian region and a high-res building or DSM height reference; then build the trained benchmark only after a clean train/val/test split.'
}
with open(OUT / 'RESULTS.json', 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2)

print('PHASE 66 STATUS:')
print(f'TRAIN_REGION: {final_results["TRAIN_REGION"]}')
print(f'VALIDATION_REGION: {final_results["VALIDATION_REGION"]}')
print(f'TEST_REGION: {final_results["TEST_REGION"]}')
print(f'REAL_OPTICAL_REGIONS: {final_results["REAL_OPTICAL_REGIONS"]}')
print(f'REAL_ELEVATION_REGIONS: {final_results["REAL_ELEVATION_REGIONS"]}')
print(f'TERRAIN_GROUND_TRUTH: {final_results["TERRAIN_GROUND_TRUTH"]}')
print(f'BUILDING_FOOTPRINT_GROUND_TRUTH: {final_results["BUILDING_FOOTPRINT_GROUND_TRUTH"]}')
print(f'BUILDING_HEIGHT_GROUND_TRUTH: {final_results["BUILDING_HEIGHT_GROUND_TRUTH"]}')
print(f'ALIGNMENT: {final_results["ALIGNMENT"]}')
print(f'GEOGRAPHIC_SPLIT: {final_results["GEOGRAPHIC_SPLIT"]}')
print(f'CURRENT_MODEL_BASELINE: {final_results["CURRENT_MODEL_BASELINE"]}')
print(f'REAL_TERRAIN_METRICS: {final_results["REAL_TERRAIN_METRICS"]}')
print(f'REAL_BUILDING_METRICS: {final_results["REAL_BUILDING_METRICS"]}')
print(f'CANNY_RESULT: {final_results["CANNY_RESULT"]}')
print(f'POINT_CLOUD_RESULT: {final_results["POINT_CLOUD_RESULT"]}')
print(f'TRAINING_READY: {final_results["TRAINING_READY"]}')
print(f'RECOMMENDED_NEXT_PHASE: {final_results["RECOMMENDED_NEXT_PHASE"]}')
