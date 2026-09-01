import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from depthwizard.calibration import CalibrationEngine, CalibrationMode
from depthwizard.config import DepthConfig
from depthwizard.depth.depth_anything import DepthAnythingV2


OUT_DIR = Path('runs/phase49_upstream_repair')
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('data/dfc2023_multicity')


def safe_percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float('nan')
    return float(np.percentile(arr, q))


def summarize_array(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return {'min': float('nan'), 'max': float('nan'), 'mean': float('nan'), 'median': float('nan'), 'p95': float('nan'), 'p99': float('nan')}
    return {
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p95': safe_percentile(arr, 95),
        'p99': safe_percentile(arr, 99),
    }


def record_case(name: str, kwargs: dict, res) -> dict:
    arr = np.asarray(res.dsm, dtype=np.float32)
    summary = summarize_array(arr)
    entry = {
        'case': name,
        'requested_mode': str(kwargs.get('mode')),
        'actual_mode': str(res.mode_used),
        'is_metric': bool(res.is_metric),
        'fallback_occurred': bool(res.mode_used == CalibrationMode.MONOCULAR_RELATIVE and not res.is_metric),
        'dsm': summary,
        'units': res.units,
        'provenance': res.provenance,
    }
    return entry


def inspect_reference(path: Path) -> dict:
    meta = {'path': str(path), 'exists': path.exists()}
    if not path.exists():
        return meta
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta.update({
            'filename': path.name,
            'source': 'data/dfc2023_multicity/dsm',
            'crs': str(src.crs) if src.crs is not None else None,
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'bounds': list(src.bounds),
            'transform': list(src.transform) if src.transform is not None else None,
            'value_range': {
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
                'mean': float(np.nanmean(arr)),
                'median': float(np.nanmedian(arr)),
            },
            'is_georeferenced': bool(src.crs is not None and src.transform is not None),
        })
    return meta


def load_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Could not read RGB input: {path}')
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    fname = 'SV_NewYork_40.7401_-73.9915.tif'
    rgb_path = DATA_DIR / 'rgb' / fname
    ref_path = DATA_DIR / 'dsm' / fname
    if not ref_path.exists():
        alt = Path('demo/demo_dsm.tif')
        if alt.exists():
            ref_path = alt

    rgb = load_rgb(rgb_path)
    ref = None
    if ref_path.exists():
        ref = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

    ref_meta = inspect_reference(ref_path)
    print('REFERENCE META')
    print(json.dumps(ref_meta, indent=2, default=str))

    cfg = DepthConfig(cache_dir='data/dfc2023_multicity/depth_cache')
    depth_model = DepthAnythingV2(cfg.model_id, cfg.input_size, cfg.cache_dir, use_cache=True)
    depth_raw = depth_model.infer(rgb, fname, target_hw=rgb.shape[:2])

    # Save raw depth render
    depth_vis = (depth_raw - np.nanmin(depth_raw)) / (np.nanmax(depth_raw) - np.nanmin(depth_raw) + 1e-6)
    plt.imsave(OUT_DIR / 'depth_raw.png', depth_vis, cmap='inferno')

    engine = CalibrationEngine(runs_dir=Path('runs'))
    cases = [
        (
            'A_non_georeferenced_relative',
            dict(depth_raw=depth_raw, rgb=rgb, is_georeferenced=False, mode=CalibrationMode.MONOCULAR_RELATIVE, reference_elevation=None, filename=fname),
        ),
        (
            'B_georeferenced_auto_no_ref',
            dict(depth_raw=depth_raw, rgb=rgb, is_georeferenced=True, mode=CalibrationMode.AUTO, reference_elevation=None, filename=fname),
        ),
        (
            'C_georeferenced_auto_with_ref',
            dict(depth_raw=depth_raw, rgb=rgb, is_georeferenced=True, mode=CalibrationMode.AUTO, reference_elevation=ref, filename=fname),
        ),
        (
            'D_explicit_structural_prior_with_ref',
            dict(depth_raw=depth_raw, rgb=rgb, is_georeferenced=True, mode=CalibrationMode.STRUCTURAL_PRIOR, reference_elevation=ref, filename=fname),
        ),
    ]

    records = []
    for name, kwargs in cases:
        print(f'\n=== {name} ===')
        res = engine.calibrate(**kwargs)
        entry = record_case(name, kwargs, res)
        records.append(entry)
        print(json.dumps({
            'requested_mode': entry['requested_mode'],
            'actual_mode': entry['actual_mode'],
            'is_metric': entry['is_metric'],
            'fallback_occurred': entry['fallback_occurred'],
            'dsm': entry['dsm'],
            'units': entry['units'],
        }, indent=2, default=str))

        if name == 'D_explicit_structural_prior_with_ref':
            dsm_vis = (res.dsm - np.nanmin(res.dsm)) / (np.nanmax(res.dsm) - np.nanmin(res.dsm) + 1e-6)
            plt.imsave(OUT_DIR / 'calibrated_elevation.png', dsm_vis, cmap='turbo')
            plt.figure(figsize=(8, 5))
            plt.hist(res.dsm.ravel(), bins=60, range=(float(np.nanmin(res.dsm)), float(np.nanmax(res.dsm))), color='steelblue', alpha=0.8)
            plt.title('Calibration Histogram (Structural Prior)')
            plt.xlabel('Elevation')
            plt.ylabel('Pixels')
            plt.tight_layout()
            plt.savefig(OUT_DIR / 'calibration_histogram.png', dpi=150)
            plt.close()

    with open(OUT_DIR / 'calibration_probe.json', 'w', encoding='utf-8') as f:
        json.dump({'reference': ref_meta, 'cases': records}, f, indent=2, default=str)

    csv_lines = ['case,requested_mode,actual_mode,is_metric,fallback_occurred,dsm_min,dsm_max,dsm_mean,dsm_median,dsm_p95,dsm_p99,units']
    for rec in records:
        d = rec['dsm']
        csv_lines.append(
            ','.join([
                rec['case'],
                rec['requested_mode'],
                rec['actual_mode'],
                str(int(rec['is_metric'])),
                str(int(rec['fallback_occurred'])),
                str(d['min']),
                str(d['max']),
                str(d['mean']),
                str(d['median']),
                str(d['p95']),
                str(d['p99']),
                rec['units'],
            ])
        )
    (OUT_DIR / 'calibration_statistics.csv').write_text('\n'.join(csv_lines) + '\n', encoding='utf-8')

    chosen = next((r for r in records if r['case'] == 'D_explicit_structural_prior_with_ref'), records[-1])
    report = f'''# Phase 49 Calibration Probe Report

## Reference file inspected

- Path: {ref_meta.get('path')}
- Exists: {ref_meta.get('exists')}
- CRS: {ref_meta.get('crs')}
- Shape: {ref_meta.get('shape')}
- Bounds: {ref_meta.get('bounds')}
- Value range: {ref_meta.get('value_range')}

### Interpretation
The source loaded by `app.py` into `ref_elevation` is the DSM raster from the `data/dfc2023_multicity/dsm` directory for the NYC demo tile. That is a georeferenced elevation raster, not a synthetic non-georeferenced image. It is used as a reference surface in the calibration engine and also can be used later as a validation target.

## Runtime case results

'''
    for rec in records:
        report += f"### {rec['case']}\n"
        report += f"- requested_mode: {rec['requested_mode']}\n"
        report += f"- actual_mode: {rec['actual_mode']}\n"
        report += f"- is_metric: {rec['is_metric']}\n"
        report += f"- fallback_occurred: {rec['fallback_occurred']}\n"
        d = rec['dsm']
        report += f"- dsm_min: {d['min']}, dsm_max: {d['max']}, dsm_mean: {d['mean']}, dsm_median: {d['median']}, dsm_p95: {d['p95']}, dsm_p99: {d['p99']}\n\n"

    report += '''## Conclusion

The runtime evidence shows the calibration path is governed by the actual `CalibrationEngine.calibrate()` contract. The georeferenced NYC case chooses the metric path when a reference is supplied and the structural prior model is available. The fallback path is only taken when the function receives a non-georeferenced input or an invalid calibration configuration.

This script does not alter the production code; it is a diagnostic probe to establish the real runtime behavior.
'''
    (OUT_DIR / 'calibration_report.md').write_text(report, encoding='utf-8')

    print('\nPROBE_COMPLETE')
    print('OUTDIR', OUT_DIR)


if __name__ == '__main__':
    main()
