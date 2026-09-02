# Phase 88: Forensic audit of Indian building height pipeline

## Objective
Forensic-only tracing of the exact Phase 87 model, tensor path, connected-component path, height capacity, and final record mapping.

TERRAIN_SOURCE = REAL GEOREFERENCED DEM
BUILDING_SOURCE = EXISTING SINGLE-VIEW BUILDING MODEL

## 1. Architecture
- Model class: BuildingConditionedEstimator / BuildingConditionedHeightNet / SmallFusionUNet.
- Parameters: 479281; trainable: 479281.
- Input: 4 channels, RGB plus relative-depth channel.
- Output: 16 feature channels plus one footprint logit.
- The model has semantic footprint segmentation and object-level height regression, but instance separation is connected-components post-processing, not a learned instance head.

## 2. Height output path
- Input -> `_prep_x` -> model forward -> footprint logit/probability -> threshold -> connected components -> model-selected object predictions -> Phase 87 per-component records.
- NAN FIRST APPEARS AT = final Phase 87 per-component building-height records.
- The model prediction list itself contains finite object heights; missing values enter when Phase 87 assigns that shorter list to the longer label-order component record list.

## 3. Component capacity audit
- UTTARAKHAND: total components=11; height-capacity=25; predicted heights=9; missing heights=2.
- HIMACHAL: total components=64; height-capacity=25; predicted heights=25; missing heights=39.

## 4. Height representation
- The actual height prediction tensor is one scalar per selected connected component, represented as a 1D tensor/list of length up to 25.
- It is not one value per pixel and not a dense height map.

## 5. Height value audit
- UTTARAKHAND: {'shape': [11], 'dtype': 'float64', 'min': 5.191013336181641, 'max': 41.42979431152344, 'mean': 17.35407288869222, 'median': 12.293121337890625, 'std': 10.971451836102299, 'p95': 35.94350280761718, 'p99': 40.33253601074219, 'finite_count': 9, 'nan_count': 2, 'nan_fraction': 0.18181818181818182, 'positive_count': 9, 'zero_count': 0, 'negative_count': 0}.
- HIMACHAL: {'shape': [64], 'dtype': 'float64', 'min': 4.402091979980469, 'max': 10.439148902893066, 'mean': 6.55735523223877, 'median': 6.321893215179443, 'std': 1.3327038618360267, 'p95': 9.08435192108154, 'p99': 10.170653114318846, 'finite_count': 25, 'nan_count': 39, 'nan_fraction': 0.609375, 'positive_count': 25, 'zero_count': 0, 'negative_count': 0}.

## 6. Building-height / footprint consistency
- Per-component flags, areas, terrain elevations, and roof elevations are preserved in COMPONENT_AUDIT.json.
- Roof elevation is computed as terrain base in meters plus predicted height in meters; no relative-depth reinterpretation is performed.

## 7. Height-footprint alignment
- UTTARAKHAND: finite mapping mismatches=9; mapping_bug_observed=True.
- HIMACHAL: finite mapping mismatches=25; mapping_bug_observed=True.
- The model sorts components by descending area before producing predictions.
- Phase 87 iterates components in connected-component label order before assigning predictions by position.
- Therefore positional correspondence is invalid whenever those orders differ.

## 8. Terrain integration
- Terrain is the Phase 72 aligned DEM in meters.
- Height outputs are existing model outputs; the audit does not claim independent Indian height accuracy.
- DSM_CANDIDATE/roof elevations are not validated against Indian building truth.

## 9. Ground truth
- INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE = NO.
- IoU, Dice, precision, recall, height MAE, height RMSE, and height correlation are not calculated.

## 10. INDIAN_DOMAIN_BEHAVIOR_COMPARISON
- Uttarakhand and Himachal distributions are recorded in DOMAIN_COMPARISON.json. These are behavior comparisons, not accuracy or generalization claims.

## 11. Final diagnosis
HEIGHT_OUTPUT_MAPPING_BUG
