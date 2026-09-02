# Phase 89: Fix building height component-ID mapping

## Objective
Surgical correction of height assignment identity using the unchanged Phase 87 inference pipeline and frozen Phase 24 checkpoint.

TERRAIN_SOURCE = REAL GEOREFERENCED DEM
BUILDING_SOURCE = EXISTING SINGLE-VIEW BUILDING MODEL

## Model component order
- The model creates connected components from the 256 x 256 thresholded footprint.
- It filters components below 16 pixels, sorts by descending area, and caps the list at 25.
- Each returned height is one scalar for one selected model component.
- Phase 89 maps each model component ID to the final 512 x 512 component ID by spatial overlap, then assigns via explicit `predicted_height_by_component_id`.

## Component audit
- UTTARAKHAND: total=11; model-selected=9; finite heights=9; missing=2; mapping mismatches after fix=0.
- HIMACHAL: total=64; model-selected=25; finite heights=25; missing=39; mapping mismatches after fix=0.

## Height statistics
- Statistics include finite predicted heights only; missing components are not replaced or interpolated.
- UTTARAKHAND: {'shape': [11], 'dtype': 'float64', 'min': 5.191013336181641, 'max': 41.42979431152344, 'mean': 17.35407288869222, 'median': 12.293121337890625, 'std': 10.971451836102299, 'p95': 35.94350280761718, 'p99': 40.33253601074219, 'finite_count': 9, 'missing_count': 2, 'missing_fraction': 0.18181818181818182}.
- HIMACHAL: {'shape': [64], 'dtype': 'float64', 'min': 4.402091979980469, 'max': 10.439148902893066, 'mean': 6.55735523223877, 'median': 6.321893215179443, 'std': 1.3327038618360265, 'p95': 9.08435192108154, 'p99': 10.170653114318846, 'finite_count': 25, 'missing_count': 39, 'missing_fraction': 0.609375}.

## Height / footprint consistency
- Every component remains in HEIGHT_ASSIGNMENTS.json with its area, selection status, height, terrain elevation, roof elevation, and flags.
- Missing-height components remain explicitly unavailable and are not assigned zero.

## Mapping validation
- Assertions passed for every finite assignment.
- Mapping mismatches after fix = 0 for Uttarakhand and Himachal.

## Terrain integration
- Terrain remains the real Phase 72 aligned DEM in meters.
- Roof elevation is terrain plus the model-predicted height in meters.
- The V2 3D prototype is visualization only and is not a validated reconstruction.

## Ground truth
- INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE = NO.
- No IoU, Dice, precision, recall, height MAE, height RMSE, or height correlation is reported.

## INDIAN_DOMAIN_BEHAVIOR_COMPARISON
- Phase 87 versus Phase 89 behavior is recorded in DOMAIN_COMPARISON.json; this is not an accuracy comparison.

## 3D prototype
- Generated as INDIAN_TERRAIN_BUILDING_3D_PROTOTYPE_V2.
- Finite-height components are extruded; missing-height components remain marked unavailable in the assignments and are not assigned height.

## Final scientific decision
HEIGHT_MAPPING_FIXED_OUTPUT_AVAILABLE_UNVALIDATED
