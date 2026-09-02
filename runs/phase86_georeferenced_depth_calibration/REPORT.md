# Phase 86 georeferenced depth calibration control

## Frozen data
- train_bbox: (1774, 2513, 2286, 3025)
- val_bbox: (5834, 7335, 6346, 7847)
- valid_train_pixels: 262144
- valid_val_pixels: 262144

## Polarity selection (train only)
- selected_sign: depth
- train_depth_vs_local_relief: {'depth': {'pearson': 0.5062006243032677, 'spearman': 0.4260707006797208}, '-depth': {'pearson': -0.5062006243032677, 'spearman': -0.42607079123433467}}

## Best local-relief model
- model_name: rank_mapping
- MAE: 465.3872039094208
- RMSE: 577.7669687438322
- Pearson: 0.05071656815432017
- Spearman: 0.07161577342289625
- mean_bias: 21.404949380530624
- prediction_std: 166.46179014730393
- target_std: 561.3603070934762
- std_ratio: 0.2965328827917738
- gradient_x_mae: nan
- gradient_y_mae: 7.3595995455500685

## Baselines
- phase82 zero baseline mae: 462.01953125
- phase82 training mean baseline mae: 462.71649169921875
- phase82 training median baseline mae: 462.01953125
- phase83 depth rank-linear control mae: 465.3872985839844

## Subsample sensitivity
- full fit mae: 465.3872039094208
- subsample fit mae: 479.1124828744214
- materially_different: False

## Interpretation
The calibration does not recover terrain structure; the selected depth-based mapping remains weakly correlated with the held-out local-relief target and does not meaningfully beat the trivial baselines.

DEPTH_CALIBRATION_INSUFFICIENT
