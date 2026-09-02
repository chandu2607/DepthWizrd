# Phase 69 Indian Terrain Training Report

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
