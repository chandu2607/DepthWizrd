# Phase 49 Calibration Probe Report

## Reference file inspected

- Path: data\dfc2023_multicity\dsm\SV_NewYork_40.7401_-73.9915.tif
- Exists: True
- CRS: EPSG:32618
- Shape: [512, 512]
- Bounds: [585148.0, 4510134.5, 585404.0, 4510390.5]
- Value range: {'min': 0.0, 'max': 87.44552612304688, 'mean': 20.745332717895508, 'median': 8.45268440246582}

### Interpretation
The source loaded by `app.py` into `ref_elevation` is the DSM raster from the `data/dfc2023_multicity/dsm` directory for the NYC demo tile. That is a georeferenced elevation raster, not a synthetic non-georeferenced image. It is used as a reference surface in the calibration engine and also can be used later as a validation target.

## Runtime case results

### A_non_georeferenced_relative
- requested_mode: CalibrationMode.MONOCULAR_RELATIVE
- actual_mode: CalibrationMode.MONOCULAR_RELATIVE
- is_metric: False
- fallback_occurred: True
- dsm_min: 0.0, dsm_max: 9.999994277954102, dsm_mean: 5.411028861999512, dsm_median: 5.4413251876831055, dsm_p95: 8.844050407409668, dsm_p99: 9.43655014038086

### B_georeferenced_auto_no_ref
- requested_mode: CalibrationMode.AUTO
- actual_mode: CalibrationMode.MONOCULAR_RELATIVE
- is_metric: False
- fallback_occurred: True
- dsm_min: 0.0, dsm_max: 9.999994277954102, dsm_mean: 5.411028861999512, dsm_median: 5.4413251876831055, dsm_p95: 8.844050407409668, dsm_p99: 9.43655014038086

### C_georeferenced_auto_with_ref
- requested_mode: CalibrationMode.AUTO
- actual_mode: CalibrationMode.STRUCTURAL_PRIOR
- is_metric: True
- fallback_occurred: False
- dsm_min: 77.36967468261719, dsm_max: 168.15499877929688, dsm_mean: 107.50129699707031, dsm_median: 107.02689361572266, dsm_p95: 131.46292114257812, dsm_p99: 152.71910095214844

### D_explicit_structural_prior_with_ref
- requested_mode: CalibrationMode.STRUCTURAL_PRIOR
- actual_mode: CalibrationMode.STRUCTURAL_PRIOR
- is_metric: True
- fallback_occurred: False
- dsm_min: 77.36967468261719, dsm_max: 168.15499877929688, dsm_mean: 107.50129699707031, dsm_median: 107.02689361572266, dsm_p95: 131.46292114257812, dsm_p99: 152.71910095214844

## Conclusion

The runtime evidence shows the calibration path is governed by the actual `CalibrationEngine.calibrate()` contract. The georeferenced NYC case chooses the metric path when a reference is supplied and the structural prior model is available. The fallback path is only taken when the function receives a non-georeferenced input or an invalid calibration configuration.

This script does not alter the production code; it is a diagnostic probe to establish the real runtime behavior.
