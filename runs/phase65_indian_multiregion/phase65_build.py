from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PH64 = ROOT / 'runs' / 'phase64_indian_data'

OUT.mkdir(parents=True, exist_ok=True)

region_a = {
    'region': 'Uttarakhand',
    'state': 'Uttarakhand',
    'coordinates': {'center': [79.55, 30.55], 'bbox': [79.35, 30.25, 79.8, 30.85]},
    'terrain_type': 'Himalayan ridge-valley terrain; steep slopes, exposed rock, settlements, roads',
    'elevation_range_m': '~1050-2300',
    'imagery_availability': 'REAL_PUBLIC_SENTINEL2_COG',
    'reference_availability': 'REAL_PUBLIC_COPERNICUS_DEM_COG',
    'optical_source': 'Sentinel-2 L2A AWS Public COG',
    'optical_resolution_m': 10,
    'elevation_source': 'Copernicus DEM GLO-30 AWS Public COG',
    'elevation_resolution_m': 30,
    'elevation_type': 'DEM',
    'building_labels': 'NO',
    'building_height_labels': 'NO',
    'slope_available': 'YES',
    'hazard_labels': 'NO',
    'CRS': 'EPSG:32644 (optical) / EPSG:4326 (DEM source)',
    'date': '2026-08-30',
    'license': 'Open public data access',
    'access': 'AWS public COG and rasterio read verified',
    'training_suitability': 'PILOT_ONLY',
    'validation_suitability': 'PENDING',
    'test_suitability': 'PENDING',
    'limitations': 'DEM is too coarse for building-height evaluation; no high-resolution building ground truth is available',
    'real_data': True,
    'metrics_available': False,
    'geographic_note': 'Validated Phase 64 pilot; not yet accepted as a final building-height benchmark.'
}

region_b = {
    'region': 'Himachal Pradesh',
    'state': 'Himachal Pradesh',
    'coordinates': {'center': [77.1, 31.9], 'bbox': [76.8, 31.5, 77.5, 32.5]},
    'terrain_type': 'Himalayan valley terrain with steep slopes, river cuts, dense settlement clusters',
    'elevation_range_m': '~1500-4200',
    'imagery_availability': 'PUBLIC_SENTINEL2_LIKELY_AVAILABLE',
    'reference_availability': 'DEM_AVAILABLE_IN_PUBLIC_CATALOGS_BUT_NOT_YET_VERIFIED',
    'optical_source': 'Sentinel-2 L2A (public STAC catalog)',
    'optical_resolution_m': 10,
    'elevation_source': 'Copernicus DEM / local DEM candidate',
    'elevation_resolution_m': 30,
    'elevation_type': 'DEM',
    'building_labels': 'CANDIDATE_ONLY',
    'building_height_labels': 'CANDIDATE_ONLY',
    'slope_available': 'YES',
    'hazard_labels': 'NO',
    'CRS': 'Scene dependent',
    'date': 'CANDIDATE',
    'license': 'Public access expected; verification pending',
    'access': 'Candidate: identified but not yet downloaded and aligned',
    'training_suitability': 'PENDING',
    'validation_suitability': 'PENDING',
    'test_suitability': 'PENDING',
    'limitations': 'No direct high-resolution building reference yet; not accepted for final evaluation',
    'real_data': False,
    'metrics_available': False,
    'geographic_note': 'Geographically separate from Phase 64 Uttarakhand and appropriate as validation candidate if a real paired tile is acquired.'
}

region_c = {
    'region': 'Sikkim',
    'state': 'Sikkim',
    'coordinates': {'center': [88.5, 27.3], 'bbox': [88.1, 27.0, 88.9, 27.6]},
    'terrain_type': 'Eastern Himalayan steep ridge and valley terrain with monsoon-driven slope dynamics',
    'elevation_range_m': '~800-5000',
    'imagery_availability': 'PUBLIC_SENTINEL2_LIKELY_AVAILABLE',
    'reference_availability': 'DEM_AVAILABLE_IN_PUBLIC_CATALOGS_BUT_NOT_YET_VERIFIED',
    'optical_source': 'Sentinel-2 L2A (public STAC catalog)',
    'optical_resolution_m': 10,
    'elevation_source': 'Copernicus DEM / local DEM candidate',
    'elevation_resolution_m': 30,
    'elevation_type': 'DEM',
    'building_labels': 'CANDIDATE_ONLY',
    'building_height_labels': 'CANDIDATE_ONLY',
    'slope_available': 'YES',
    'hazard_labels': 'NO',
    'CRS': 'Scene dependent',
    'date': 'CANDIDATE',
    'license': 'Public access expected; verification pending',
    'access': 'Candidate: identified but not yet downloaded and aligned',
    'training_suitability': 'PENDING',
    'validation_suitability': 'PENDING',
    'test_suitability': 'PENDING',
    'limitations': 'No direct high-resolution building reference yet; not accepted for final evaluation',
    'real_data': False,
    'metrics_available': False,
    'geographic_note': 'Geographically separated from both Uttarakhand and Himachal; valid as a final test candidate only if a trusted paired benchmark is acquired.'
}

regions = [region_a, region_b, region_c]

matrix_rows = []
for r in regions:
    row = {
        'region': r['region'],
        'state': r['state'],
        'optical_source': r['optical_source'],
        'optical_resolution': r['optical_resolution_m'],
        'elevation_source': r['elevation_source'],
        'elevation_resolution': r['elevation_resolution_m'],
        'elevation_type': r['elevation_type'],
        'building_labels': r['building_labels'],
        'building_height_labels': r['building_height_labels'],
        'slope_available': r['slope_available'],
        'hazard_labels': r['hazard_labels'],
        'CRS': r['CRS'],
        'date': r['date'],
        'license': r['license'],
        'access': r['access'],
        'training_suitability': r['training_suitability'],
        'validation_suitability': r['validation_suitability'],
        'test_suitability': r['test_suitability'],
        'limitations': r['limitations'],
    }
    matrix_rows.append(row)

with open(OUT / 'INDIAN_DATASET_MATRIX.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
    writer.writeheader()
    writer.writerows(matrix_rows)

manifest_rows = [
    {
        'split': 'TRAIN',
        'region': region_a['region'],
        'state': region_a['state'],
        'coordinates': str(region_a['coordinates']['bbox']),
        'real_data': 'YES',
        'status': 'REAL_PILOT_BUT_NOT_HIGH_RES_BUILDING_GT',
        'leakage_risk': 'LOW',
        'notes': 'Valid physical benchmark pair exists; no accepted building-height ground truth is available.'
    },
    {
        'split': 'VALIDATION',
        'region': region_b['region'],
        'state': region_b['state'],
        'coordinates': str(region_b['coordinates']['bbox']),
        'real_data': 'NO',
        'status': 'CANDIDATE_ONLY',
        'leakage_risk': 'LOW',
        'notes': 'Geographically separate from A; not yet downloaded and aligned.'
    },
    {
        'split': 'TEST',
        'region': region_c['region'],
        'state': region_c['state'],
        'coordinates': str(region_c['coordinates']['bbox']),
        'real_data': 'NO',
        'status': 'CANDIDATE_ONLY',
        'leakage_risk': 'LOW',
        'notes': 'Geographically separate from A and B; not yet downloaded and aligned.'
    }
]

with open(OUT / 'INDIA_SPLIT_MANIFEST.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['split', 'region', 'state', 'coordinates', 'real_data', 'status', 'leakage_risk', 'notes'])
    writer.writeheader()
    writer.writerows(manifest_rows)

phase64_status = json.loads((PH64 / 'RESULTS.json').read_text(encoding='utf-8')) if (PH64 / 'RESULTS.json').exists() else {}
phase64_status.setdefault('PHASE_64_STATUS', 'REAL_BENCHMARK_ESTABLISHED')

with open(OUT / 'REGION_A_METADATA.json', 'w', encoding='utf-8') as f:
    json.dump(region_a, f, indent=2)
with open(OUT / 'REGION_B_METADATA.json', 'w', encoding='utf-8') as f:
    json.dump(region_b, f, indent=2)
with open(OUT / 'REGION_C_METADATA.json', 'w', encoding='utf-8') as f:
    json.dump(region_c, f, indent=2)

with open(OUT / 'DATASET_SELECTION.md', 'w', encoding='utf-8') as f:
    f.write('# Phase 65 — Indian Multi-Region Dataset Selection\n\n')
    f.write('## Decision\n\n')
    f.write('The project currently has one verified real Indian mountainous benchmark pair from Phase 64: Uttarakhand.\n\n')
    f.write('A second and third region were identified as geographically separate candidates: Himachal Pradesh and Sikkim. These are scientifically appropriate for train/validation/test separation, but they are not yet accepted as final benchmark regions because the paired optical + elevation files have not yet been downloaded, read, aligned, and matched with a valid high-resolution building or height reference.\n\n')
    f.write('## Scientific position\n\n')
    f.write('- No Indian train/validation/test split is accepted yet.\n')
    f.write('- Phase 64 remains the only verified real Indian benchmark pair.\n')
    f.write('- No high-resolution building-height ground truth has been verified for any Indian region in this workspace.\n')
    f.write('- The project stops before training, exactly as required by the evidence gate.\n')

alignment_report = '''# Phase 65 — Alignment Status\n\n## Verified today\n\n- Region A (Uttarakhand): real optical file and real DEM were downloaded, read, and aligned in Phase 64.\n- Region B (Himachal): identified as a candidate, but no real raster pair has yet been downloaded and aligned.\n- Region C (Sikkim): identified as a candidate, but no real raster pair has yet been downloaded and aligned.\n\n## Alignment conclusion\n\nThe project has not yet achieved a fully accepted three-region Indian benchmark. Region A is real and aligned; B and C are not yet accepted benchmark pairs.\n'''
with open(OUT / 'ALIGNMENT_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(alignment_report)

quality_report = '''# Phase 65 — Data Quality Report\n\n## Region A: Uttarakhand\n\n- Real public optical file: YES\n- Real public elevation file: YES\n- Raster read test: YES\n- CRS check: YES\n- Alignment check: YES\n- High-resolution building-height reference: NO\n\n## Region B: Himachal Pradesh\n\n- Candidate only; real raster pair not yet downloaded and validated.\n\n## Region C: Sikkim\n\n- Candidate only; real raster pair not yet downloaded and validated.\n\n## Conclusion\n\nThe benchmark quality gate is not yet satisfied for a three-region India-ready training set.\n'''
with open(OUT / 'DATA_QUALITY_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(quality_report)

baseline_headers = ['dataset', 'task', 'checkpoint', 'checkpoint_sha256', 'preprocessing', 'input_dimensions', 'output_dimensions', 'runtime_sec', 'MAE', 'RMSE', 'bias', 'correlation', 'notes']
baseline_rows = [{
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
    'notes': 'Current production model applied unchanged to real Indian scene; no accepted metric reference.'
}]
with open(OUT / 'BASELINE_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=baseline_headers)
    writer.writeheader()
    writer.writerows(baseline_rows)

terrain_rows = [
    {'region': 'Uttarakhand', 'elevation_source': 'Copernicus DEM GLO-30', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'slope_band': 'ALL', 'notes': 'Real DEM exists but no accepted high-resolution reference for building-scale evaluation.'},
    {'region': 'Himachal Pradesh', 'elevation_source': 'CANDIDATE', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'slope_band': 'PENDING', 'notes': 'Candidate only; no real aligned pair yet.'},
    {'region': 'Sikkim', 'elevation_source': 'CANDIDATE', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'slope_band': 'PENDING', 'notes': 'Candidate only; no real aligned pair yet.'},
]
with open(OUT / 'TERRAIN_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region', 'elevation_source', 'MAE', 'RMSE', 'bias', 'correlation', 'slope_band', 'notes'])
    writer.writeheader()
    writer.writerows(terrain_rows)

building_rows = [
    {'region': 'Uttarakhand', 'IoU': 'NOT_AVAILABLE', 'Dice': 'NOT_AVAILABLE', 'Precision': 'NOT_AVAILABLE', 'Recall': 'NOT_AVAILABLE', 'Boundary_F1': 'NOT_AVAILABLE', 'Instance_matching': 'NOT_AVAILABLE', 'Building_Height_MAE': 'NOT_AVAILABLE', 'Building_Height_RMSE': 'NOT_AVAILABLE', 'Bias': 'NOT_AVAILABLE', 'notes': 'No valid building-height labels available.'},
    {'region': 'Himachal Pradesh', 'IoU': 'NOT_AVAILABLE', 'Dice': 'NOT_AVAILABLE', 'Precision': 'NOT_AVAILABLE', 'Recall': 'NOT_AVAILABLE', 'Boundary_F1': 'NOT_AVAILABLE', 'Instance_matching': 'NOT_AVAILABLE', 'Building_Height_MAE': 'NOT_AVAILABLE', 'Building_Height_RMSE': 'NOT_AVAILABLE', 'Bias': 'NOT_AVAILABLE', 'notes': 'Candidate only; no accepted reference.'},
    {'region': 'Sikkim', 'IoU': 'NOT_AVAILABLE', 'Dice': 'NOT_AVAILABLE', 'Precision': 'NOT_AVAILABLE', 'Recall': 'NOT_AVAILABLE', 'Boundary_F1': 'NOT_AVAILABLE', 'Instance_matching': 'NOT_AVAILABLE', 'Building_Height_MAE': 'NOT_AVAILABLE', 'Building_Height_RMSE': 'NOT_AVAILABLE', 'Bias': 'NOT_AVAILABLE', 'notes': 'Candidate only; no accepted reference.'},
]
with open(OUT / 'BUILDING_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region', 'IoU', 'Dice', 'Precision', 'Recall', 'Boundary_F1', 'Instance_matching', 'Building_Height_MAE', 'Building_Height_RMSE', 'Bias', 'notes'])
    writer.writeheader()
    writer.writerows(building_rows)

slope_rows = [
    {'region': 'Uttarakhand', 'slope_band_deg': '0-5', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'Slope available from DEM crop, but no accepted ground-truth metric reference yet.'},
    {'region': 'Uttarakhand', 'slope_band_deg': '5-15', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'Slope available from DEM crop, but no accepted ground-truth metric reference yet.'},
    {'region': 'Uttarakhand', 'slope_band_deg': '15-30', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'Slope available from DEM crop, but no accepted ground-truth metric reference yet.'},
    {'region': 'Uttarakhand', 'slope_band_deg': '30-45', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'Slope available from DEM crop, but no accepted ground-truth metric reference yet.'},
    {'region': 'Uttarakhand', 'slope_band_deg': '>45', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'Slope available from DEM crop, but no accepted ground-truth metric reference yet.'},
    {'region': 'Himachal Pradesh', 'slope_band_deg': 'PENDING', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'Not yet acquired.'},
    {'region': 'Sikkim', 'slope_band_deg': 'PENDING', 'MAE': 'NOT_AVAILABLE', 'RMSE': 'NOT_AVAILABLE', 'bias': 'NOT_AVAILABLE', 'correlation': 'NOT_AVAILABLE', 'notes': 'Not yet acquired.'},
]
with open(OUT / 'SLOPE_STRATIFIED_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region', 'slope_band_deg', 'MAE', 'RMSE', 'bias', 'correlation', 'notes'])
    writer.writeheader()
    writer.writerows(slope_rows)

canny_rows = [{
    'region': 'Uttarakhand',
    'canny_role': 'EXPERIMENTAL_ONLY',
    'boundary_help': 'UNVERIFIED',
    'terrain_false_edges': 'YES',
    'vegetation_false_edges': 'YES',
    'road_false_edges': 'YES',
    'notes': 'Canny remains experimental; it is not integrated into production.'
}]
with open(OUT / 'CANNY_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region', 'canny_role', 'boundary_help', 'terrain_false_edges', 'vegetation_false_edges', 'road_false_edges', 'notes'])
    writer.writeheader()
    writer.writerows(canny_rows)

pointcloud_rows = [{
    'region': 'Uttarakhand',
    'point_cloud_role': 'EXPERIMENTAL_ONLY',
    'continuity': 'UNVERIFIED',
    'height_preservation': 'UNVERIFIED',
    'terrain_smoothness': 'UNVERIFIED',
    'spikes': 'UNVERIFIED',
    'holes': 'UNVERIFIED',
    'notes': 'Point cloud is a pilot analysis only; it does not replace the production renderer.'
}]
with open(OUT / 'POINT_CLOUD_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['region', 'point_cloud_role', 'continuity', 'height_preservation', 'terrain_smoothness', 'spikes', 'holes', 'notes'])
    writer.writeheader()
    writer.writerows(pointcloud_rows)

training_readiness = '''# Phase 65 — Training Readiness\n\n## Decision\n\nTRAINING_READY = NO\n\n## Why not\n\n1. Only one real Indian region has been downloaded and aligned at acceptable scientific quality: Uttarakhand.\n2. A second and third geographically separated region have been identified but not yet acquired and validated.\n3. No high-resolution building reference or building-height labels have been verified for the Indian benchmark.\n4. The project has not yet established a valid train/validation/test split with leakage-safe geographic separation.\n\n## Fit-for-future options\n\nThe next stage should not be training. It should be: (A) acquire and verify a second and third valid Indian region; (B) secure a proper high-resolution building/height reference; (C) then decide between global + India mixed training, terrain-balanced training, or a dual-branch terrain/buildings pipeline based on baseline failure analysis.\n'''
with open(OUT / 'TRAINING_READINESS.md', 'w', encoding='utf-8') as f:
    f.write(training_readiness)

report_md = '''# Phase 65 — Indian Multi-Region Benchmark Report\n\n## Summary\n\nThis phase preserves the verified Phase 64 Uttarakhand evidence and builds the scientific framework for a multi-region Indian dataset. The goal is not to train or modify production. The goal is to establish proof that a valid Indian benchmark can exist before any training decision is made.\n\n## Verified status\n\n- Real Indian optical + elevation benchmark exists for Region A (Uttarakhand): YES\n- Real benchmark bytes read successfully: YES\n- High-resolution building-height ground truth: NO\n- Second region acquired: NO\n- Third region acquired: NO\n- Training started: NO\n- Architecture changed: NO\n\n## Scientific conclusion\n\nA valid India-capable training pipeline is not ready yet. The project has evidence for one real benchmark pair and a credible multi-region selection plan, but it does not yet have the required three-region benchmark package needed for a scientifically defensible Indian training study.\n'''
with open(OUT / 'REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report_md)

results = {
    'PHASE_65_STATUS': 'MULTI_REGION_BENCHMARK_FRAMEWORK_CREATED',
    'TRAIN_REGION': 'Uttarakhand (pilot benchmark)',
    'VALIDATION_REGION': 'Himachal Pradesh (candidate; not yet real benchmark)',
    'TEST_REGION': 'Sikkim (candidate; not yet real benchmark)',
    'REAL_OPTICAL_REGIONS': 1,
    'REAL_ELEVATION_REGIONS': 1,
    'HIGH_RES_BUILDING_REFERENCE': 'NO',
    'HIGH_RES_HEIGHT_REFERENCE': 'NO',
    'GEOGRAPHIC_LEAKAGE_CHECK': 'NOT_YET_ACCEPTED',
    'ALIGNMENT_STATUS': 'A_VERIFIED_ONLY; B_AND_C_UNVERIFIED',
    'CURRENT_MODEL_BASELINE': 'YES_ON_REAL_UTTARAKHAND',
    'REAL_METRICS_AVAILABLE': 'NO',
    'CANNY_ROLE': 'EXPERIMENTAL_ONLY',
    'POINT_CLOUD_ROLE': 'EXPERIMENTAL_ONLY',
    'TRAINING_READY': 'NO',
    'TRAINING_STARTED': 'NO',
    'ARCHITECTURE_CHANGED': 'NO',
    'phase64_reference': str(PH64 / 'RESULTS.json')
}
with open(OUT / 'RESULTS.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print('PHASE 65 STATUS:')
print(f'TRAIN_REGION: {results["TRAIN_REGION"]}')
print(f'VALIDATION_REGION: {results["VALIDATION_REGION"]}')
print(f'TEST_REGION: {results["TEST_REGION"]}')
print(f'REAL_OPTICAL_REGIONS: {results["REAL_OPTICAL_REGIONS"]}')
print(f'REAL_ELEVATION_REGIONS: {results["REAL_ELEVATION_REGIONS"]}')
print(f'HIGH_RES_BUILDING_REFERENCE: {results["HIGH_RES_BUILDING_REFERENCE"]}')
print(f'HIGH_RES_HEIGHT_REFERENCE: {results["HIGH_RES_HEIGHT_REFERENCE"]}')
print(f'GEOGRAPHIC_LEAKAGE_CHECK: {results["GEOGRAPHIC_LEAKAGE_CHECK"]}')
print(f'ALIGNMENT_STATUS: {results["ALIGNMENT_STATUS"]}')
print(f'CURRENT_MODEL_BASELINE: {results["CURRENT_MODEL_BASELINE"]}')
print(f'REAL_METRICS_AVAILABLE: {results["REAL_METRICS_AVAILABLE"]}')
print(f'CANNY_ROLE: {results["CANNY_ROLE"]}')
print(f'POINT_CLOUD_ROLE: {results["POINT_CLOUD_ROLE"]}')
print(f'TRAINING_READY: {results["TRAINING_READY"]}')
print('TRAINING_STARTED: NO')
print('ARCHITECTURE_CHANGED: NO')
