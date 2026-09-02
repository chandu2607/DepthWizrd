# Phase 82 controlled terrain learning trajectory

## Setup
- Exact same frozen Phase 72 Uttarakhand data, split, local-relief target, TerrainUNet, optimizer, LR, masks, and normalization as Phase 79.
- Only variation: 5 epochs, same fixed Uttarakhand validation region, same valid-pixel-only evaluation.

## Baselines (fixed, not recomputed from validation)
- zero_baseline: MAE=462.019531, RMSE=563.863953, Pearson=nan, prediction_mean=0.000000, prediction_std=0.000000, target_std=561.360291, mean_bias=53.076824
- training_mean_baseline: MAE=462.716492, RMSE=561.768250, Pearson=nan, prediction_mean=-31.671877, prediction_std=0.000006, target_std=561.360291, mean_bias=21.404945
- training_median_baseline: MAE=462.019531, RMSE=563.864014, Pearson=nan, prediction_mean=0.000244, prediction_std=0.000000, target_std=561.360291, mean_bias=53.077068

## Epoch history
- epoch 1: train_loss=0.448941, val_loss=0.799093, val_mae=1.223687, val_rmse=1.476800, pearson=-0.057983660671387625, pred_mean=-0.113920, pred_std=0.011822, target_std=1.474941, mean_bias=-0.057679, grad_x_mae=0.016214, grad_y_mae=0.010780, std_ratio=0.008015, param_rel_change=0.161383
- epoch 2: train_loss=0.443670, val_loss=0.798705, val_mae=1.222765, val_rmse=1.477790, pearson=-0.07768293867167185, pred_mean=-0.103163, pred_std=0.024485, target_std=1.474941, mean_bias=-0.046923, grad_x_mae=0.016160, grad_y_mae=0.010676, std_ratio=0.016601, param_rel_change=0.118883
- epoch 3: train_loss=0.438470, val_loss=0.798555, val_mae=1.221865, val_rmse=1.479978, pearson=-0.07982575669335891, pred_mean=-0.092695, pred_std=0.047844, target_std=1.474941, mean_bias=-0.036455, grad_x_mae=0.016074, grad_y_mae=0.010523, std_ratio=0.032438, param_rel_change=0.088267
- epoch 4: train_loss=0.429948, val_loss=0.800366, val_mae=1.222192, val_rmse=1.487238, pearson=-0.09377922577977701, pred_mean=-0.073417, pred_std=0.096762, target_std=1.474941, mean_bias=-0.017176, grad_x_mae=0.015929, grad_y_mae=0.010269, std_ratio=0.065604, param_rel_change=0.186417
- epoch 5: train_loss=0.412883, val_loss=0.812841, val_mae=1.231205, val_rmse=1.516950, pearson=-0.11710258502862944, pred_mean=-0.013261, pred_std=0.219291, target_std=1.474941, mean_bias=0.042980, grad_x_mae=0.015722, grad_y_mae=0.010024, std_ratio=0.148678, param_rel_change=0.071492

## Interpretation
- Max Pearson across 5 epochs: -0.057983660671387625
- Final Pearson: -0.11710258502862944
- Final prediction std / target std: 0.148678
- Final gradient-X MAE: 0.015722
- Final gradient-Y MAE: 0.010024

## Final label
MULTI_EPOCH_MODEL_COLLAPSE
