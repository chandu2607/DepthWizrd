# Phase 79 spatial terrain baseline

## Controlled setup
- Same Phase 72 RGB/DEM/mask grid and the same valid-mask crop logic as Phase 77.
- Same Phase 75 local-relief target definition: DEM - median(valid DEM pixels in crop).
- Same Phase 77 spatial train/validation strategy with a single Uttarakhand training crop and one held-out Uttarakhand validation crop.
- Only model change: TerrainHead -> TerrainUNet. No other training or data changes.

## Input verification
- Input tensor shape: [4, 512, 512]
- Input dtype: torch.float32
- Input min/max/mean/std: -1.5172373056411743, 3.397331476211548, 0.002012006938457489, 0.5000823736190796
- Finite count: 1048576

## Baseline comparison
- zero_relief_baseline: MAE=1.2158, RMSE=1.4760, Pearson=nan, prediction_mean=0.0000, prediction_std=0.0000, target_std=1.4749, mean_bias=0.0562
- mean_relief_baseline: MAE=1.2191, RMSE=1.4749, Pearson=nan, prediction_mean=-0.0562, prediction_std=0.0000, target_std=1.4749, mean_bias=0.0000
- median_relief_baseline: MAE=1.2139, RMSE=1.4815, Pearson=nan, prediction_mean=0.0832, prediction_std=0.0000, target_std=1.4749, mean_bias=0.1395
- terrain_unet: MAE=1.2237, RMSE=1.4768, Pearson=-0.057983660671387625, prediction_mean=-0.1139, prediction_std=0.0118, target_std=1.4749, mean_bias=-0.0577

## Spatial diagnostics
- gradient-X MAE: 0.016214
- gradient-Y MAE: 0.010780
- prediction gradient magnitude mean: 0.000116
- target gradient magnitude mean: 0.013600

## Sensitivity control
- original prediction std: 0.001941
- shuffled prediction std: 0.002382
- constant prediction std: 0.000219
- original gradient magnitude mean: 0.000080
- shuffled gradient magnitude mean: 0.005156
- constant gradient magnitude mean: 0.000020

## Old vs new comparison
- old TerrainHead MAE: 463.506775
- old TerrainHead RMSE: 570.499573
- old TerrainHead Pearson: 0.064182
- old TerrainHead prediction_std: 1.436009
- new TerrainUNet MAE: 1.223687
- new TerrainUNet RMSE: 1.476800
- new TerrainUNet Pearson: -0.057983660671387625
- new TerrainUNet prediction_std: 0.011822

Decision: SPATIAL_TERRAIN_REGRESSION_STILL_FAILS

SPATIAL_TERRAIN_REGRESSION_STILL_FAILS
