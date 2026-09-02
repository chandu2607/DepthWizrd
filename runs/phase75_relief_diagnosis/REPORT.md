# Phase 75 relief diagnosis

## Control setup
- Model: same as Phase 73 (SmallFusionUNet with frozen Depth Anything V2 prior).
- Training signal: absolute DEM target vs local-relief target.
- Regions: Uttarakhand train, Himachal validation; Sikkim not evaluated.
- Training length: one epoch only, same optimizer and learning rate as Phase 73.
- Loss: SmoothL1Loss, same as Phase 73.

## Target proof
- Uttarakhand absolute DEM: min=3472.830078 m, max=5375.153320 m, mean=4662.035156 m, std=355.376282 m, median=4687.724121 m.
- Uttarakhand local relief: min=-1214.894043 m, max=687.429199 m, mean=-2.568882e+01 m, std=355.376282 m.
- Himachal absolute DEM: min=255.247849 m, max=283.818054 m, mean=260.328979 m, std=2.054397 m, median=259.963684 m.
- Himachal local relief: min=-4.715836 m, max=23.854370 m, mean=3.652686e-01 m, std=2.054397 m.

## Local-relief one-epoch validation result
- Prediction min/max/mean/std: 47.161953 m / 51.243305 m / 49.569866 m / 0.312934 m.
- Target min/max/mean/std: -4.715836 m / 23.854366 m / 0.365269 m / 2.054397 m.
- MAE = 49.204597 m.
- RMSE = 49.247993 m.
- Pearson = 0.035733.

## Explicit comparison
- Phase 73 absolute target: MAE=4476.964844 m, RMSE=4476.965332 m, correlation=0.035734, prediction mean=4623.635254 m.
- Phase 75 local-relief target: MAE=49.204597 m, RMSE=49.247993 m, correlation=0.035733, prediction mean=49.569866 m.

## Decision: DIAGNOSIS_INCONCLUSIVE

DIAGNOSIS_INCONCLUSIVE
