# Phase 76 relief-learning diagnosis

## Frozen inputs
- TRAIN = Uttarakhand
- VALIDATION = Himachal
- Sikkim remains locked and was not evaluated.

## Baselines
- ZERO RELIEF: MAE=1.177090 m, RMSE=2.086616 m, Pearson=nan
- MEAN RELIEF: MAE=26.054083 m, RMSE=26.134953 m, Pearson=nan
- MEDIAN RELIEF: MAE=1.177090 m, RMSE=2.086616 m, Pearson=nan

## Spatial-shuffle control
- Shuffled prediction: MAE=49.204598 m, RMSE=49.248442 m, Pearson=0.001420498439276439 (seed=1337)

## Constant-prediction check
- Constant prediction vs true relief: MAE=1.177090 m, RMSE=2.086616 m, Pearson=nan
- The metric implementation explicitly returns NaN when the prediction variance is zero because the correlation is undefined in that case.

## Spatial structure
- Valid-pixel correlation: 0.035733
- MAE: 49.204597 m
- RMSE: 49.247993 m
- Mean bias: 49.204597 m
- Prediction std: 0.312934 m
- Target std: 2.054397 m
- Error std: 2.067010 m
- Mean absolute gradient difference: 0.167681

## Phase 75 vs baselines
- Phase 75 local relief: MAE=49.204597 m, RMSE=49.247993 m, Pearson=0.035733, prediction mean=49.569866 m
- Best baseline: zero=1.177090 m, mean=26.054083 m, median=1.177090 m

## Decision: LOCAL_RELIEF_IMPROVEMENT_NOT_YET_DEMONSTRATED

LOCAL_RELIEF_IMPROVEMENT_NOT_YET_DEMONSTRATED
