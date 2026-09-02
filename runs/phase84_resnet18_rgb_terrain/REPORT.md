# Phase 84 ResNet-18 RGB terrain regression baseline

INPUT = RGB ONLY
DEPTH = NOT USED
TARGET = LOCAL RELIEF
TRAIN = UTTARAKHAND
VALIDATION = HELD-OUT UTTARAKHAND
TEST = SIKKIM LOCKED

## Pretrained encoder status
- weights_loaded: True
- weights_identifier: IMAGENET1K_V1
- parameter_count: 14882561
- trainable_parameter_count: 14882561
- encoder_trainable: True

## Data and target summary
- raw relief min/max/mean/std: -803.090332/881.490234/-31.671883/380.598511
- normalized relief min/max/mean/std: -2.026856/2.399279/0.000000/1.000000
- normalization mean/std: -31.671883/380.598511

## Input summary
- shape: [1, 3, 512, 512]
- dtype: torch.float32
- min/max/mean/std: 0.000000/0.211307/0.042897/0.063170
- finite_count: 786432

## Loss verification
- implementation_loss: 0.000000000000
- reference_loss: 0.000000000000
- absolute_difference: 0.000000000000

## Feature shapes
- layer1: [1, 64, 128, 128]
- layer2: [1, 128, 64, 64]
- layer3: [1, 256, 32, 32]
- layer4: [1, 512, 16, 16]

## Epoch history
- epoch 1: train_loss=0.445333, val_loss=0.795789, meter_mae=463.545074, meter_rmse=561.714478, meter_pearson=-0.10500871122054445, meter_pred_std=2.540356, meter_target_std=561.360291, std_ratio=0.004525, mean_bias=9.580887, grad_x_mae=6.066587, grad_y_mae=4.038424
- epoch 2: train_loss=0.421026, val_loss=0.866082, meter_mae=486.835754, meter_rmse=603.785034, meter_pearson=-0.3448284267619926, meter_pred_std=38.793938, meter_target_std=561.360291, std_ratio=0.069107, mean_bias=181.403198, grad_x_mae=5.740283, grad_y_mae=3.881450
- epoch 3: train_loss=0.637301, val_loss=0.797311, meter_mae=462.831024, meter_rmse=564.836426, meter_pearson=-0.14227346721400913, meter_pred_std=5.726273, meter_target_std=561.360291, std_ratio=0.010201, mean_bias=54.473434, grad_x_mae=5.985948, grad_y_mae=4.045795
- epoch 4: train_loss=0.335289, val_loss=0.796159, meter_mae=463.625458, meter_rmse=562.026550, meter_pearson=-0.05360315251411467, meter_pred_std=8.273611, meter_target_std=561.360291, std_ratio=0.014739, mean_bias=13.492882, grad_x_mae=5.927426, grad_y_mae=3.944683
- epoch 5: train_loss=0.296363, val_loss=0.809713, meter_mae=470.979797, meter_rmse=565.598450, meter_pearson=0.012977222802291478, meter_pred_std=26.111032, meter_target_std=561.360291, std_ratio=0.046514, mean_bias=-66.894524, grad_x_mae=5.764584, grad_y_mae=3.831575

## Phase 82 vs Phase 84 comparison
- Phase 82 TerrainUNet: MAE=1.231205, RMSE=1.516950, Pearson=-0.11710258502862944, prediction_std=0.219291
- Phase 84 TerrainResNet18: MAE=470.979797, RMSE=565.598450, Pearson=0.012977222802291478, prediction_std=26.111032

## Final label
RESNET18_DOES_NOT_SOLVE_TERRAIN_COLLAPSE
