# Phase 83 depth-only terrain structure control

## Data and validation setup
- Same held-out Uttarakhand validation split as Phase 77/82.
- Exact Phase 78 depth generation path from Depth Anything V2 raw relative-depth inference.
- No Sikkim evaluation; no retraining; no production edits.

## Raw depth summary
- train depth min/max/mean/std: 1.139898/3.164726/1.765007/0.412004
- validation depth min/max/mean/std: 1.139898/3.164726/1.765007/0.412004
- validation finite: True
- validation non-constant: True
- validation spatial variation: True

## Depth-to-terrain correlation
- Pearson(depth, local_relief): -0.04026397787777201
- Spearman(depth, local_relief): 0.07161557140858296
- Pearson(-depth, local_relief): 0.04026397787777201
- Spearman(-depth, local_relief): -0.07161557545229966

## Deterministic depth controls
- A normalized depth: MAE=539.544312, RMSE=691.117798, Pearson=-0.04026397995808913, prediction_std=380.598511, gradient_x_mae=4.968737, gradient_y_mae=2.732150
- B inverted normalized depth: MAE=564.160645, RMSE=665.758057, Pearson=0.040263975535705876, prediction_std=380.598511, gradient_x_mae=4.968737, gradient_y_mae=2.732150
- C rank-linear depth mapping: MAE=465.387299, RMSE=577.767151, Pearson=0.050716245778952625, a=576.639954, b=-319.991821, prediction_std=166.462265, gradient_x_mae=5.594278, gradient_y_mae=3.000343

## Comparison against Phase 82 metrics
- DEPTH_CONTROL (C_rank_linear_depth_mapping): MAE=465.387299, RMSE=577.767151, Pearson=0.050716245778952625, prediction_std=166.462265, gradient_x_mae=5.594278, gradient_y_mae=3.000343
- TERRAINUNET: MAE=1.231205, RMSE=1.516950, Pearson=-0.11710258502862944, prediction_std=0.219291, gradient_x_mae=0.015722, gradient_y_mae=0.010024
- ZERO_BASELINE: MAE=462.019531, RMSE=563.863953, Pearson=nan, prediction_std=0.000000
- TRAINING_MEAN_BASELINE: MAE=462.716492, RMSE=561.768250, Pearson=nan, prediction_std=0.000006
- TRAINING_MEDIAN_BASELINE: MAE=462.019531, RMSE=563.864014, Pearson=nan, prediction_std=0.000000

## Interpretation
- Best deterministic depth control: C_rank_linear_depth_mapping
- Final decision: DEPTH_PRIOR_INSUFFICIENT_FOR_TERRAIN

## Final label
DEPTH_PRIOR_INSUFFICIENT_FOR_TERRAIN
