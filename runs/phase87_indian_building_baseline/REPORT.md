# Phase 87: Indian mountain building height baseline

## 1. Objective
Test whether the existing building-oriented optical model produces structurally usable building outputs on real Indian mountainous optical imagery while the real georeferenced DEM supplies the terrain base.

TERRAIN_SOURCE = REAL_GEOREFERENCED_DEM
BUILDING_SOURCE = SINGLE_VIEW_OPTICAL_MODEL

## 2. Indian data used
- Development: Uttarakhand.
- Validation: Himachal Pradesh.
- Sikkim: not evaluated; locked because verified Indian building ground truth is unavailable.
- Optical: real Sentinel-2 B04/B03/B02 composites from the frozen Phase 68 source.
- Crop policy: deterministic center 512 x 512 crop per region.

## 3. Existing building model used
- Model: `BuildingConditionedEstimator -> BuildingConditionedHeightNet -> SmallFusionUNet`.
- Checkpoint: `C:\Users\chand\OneDrive\Desktop\DepthWizard\runs\phase24_moe\seed_0\model.pt`.
- Reused without retraining or threshold tuning.
- Input: 4 channels, RGB plus normalized relative depth.
- Footprint threshold: inherited sigmoid probability threshold 0.5.
- Height output: existing object-level model output; not ground truth.

## 4. Terrain DEM information
- Phase 72 aligned DEM used only as the metric terrain base.
- DEM was not passed into the building detector.

## 5. Building inference results
### Uttarakhand
- Foreground: 151425 pixels (57.7641%).
- Probability range: 0.07621994614601135 to 0.9935264587402344.
- Components after area filter: 11.
- Largest component ratio: 0.8454.
- Detected buildings after post-processing: 11.
### Himachal Pradesh
- Foreground: 44449 pixels (16.9559%).
- Probability range: 0.11338089406490326 to 0.9477707743644714.
- Components after area filter: 64.
- Largest component ratio: 0.4157.
- Detected buildings after post-processing: 64.

## 6. Height inference results
- Heights below are MODEL OUTPUT, not accuracy against Indian building-height truth.
- Uttarakhand: min=5.191013336181641, max=41.42979431152344, mean=17.35407257080078, std=10.971451759338379 m.
- Himachal Pradesh: min=4.402091979980469, max=10.439148902893066, mean=6.557355880737305, std=1.3327038288116455 m.

## 7. Ground-truth availability
- INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE = NO.
- No IoU, Dice, precision, recall, height MAE, height RMSE, or height correlation is reported.

## 8. Domain-shift observations
- Indian-only output distributions are recorded in DOMAIN_COMPARISON.json.
- Existing non-Indian results are not numerically comparable because the imagery, labels, resolution, and scene regime differ.

## 9. Failure flags
- Uttarakhand: NAN_HEIGHTS.
- Himachal Pradesh: NAN_HEIGHTS.

## 10. 3D prototype status
- No 3D renderer was run in this forensic baseline.
- The diagnostic views show the conceptual DSM_CANDIDATE only; it is not validated 3D reconstruction.

## 11. Limitations
- One crop per region cannot establish regional generalization.
- No verified Indian building labels exist in the repository.
- Optical imagery is 10 m resolution, so small buildings may be unresolved.
- Predicted height distributions may reflect domain shift and must not be interpreted as measured heights.

## 12. Final scientific decision
INDIAN_BUILDING_OUTPUT_AVAILABLE_BUT_UNVALIDATED
