# PHASE 9 - MULTI-CITY TRAINING EXPERIMENT

## Hypothesis
Broader training across multiple cities exposes the model to more diverse depth->height relationships and improves unseen-city generalization.

## Exposure
- Arm A >15m px: 11746664, >30m: 3160640
- Arm B >15m px: 27406002, >30m: 6645375

## Results (Mean ± Std over Seeds)
| Metric | Arm A | Arm B |
|---|---|---|
| all_mae_pooled | 11.721±0.072 | 11.992±0.028 |
| all_rmse_pooled | 20.793±0.235 | 21.337±0.041 |
| building_mae_pooled | 26.498±0.318 | 27.437±0.071 |
| building_rmse_pooled | 31.466±0.373 | 32.346±0.064 |
| building_bias_mean | n/a | n/a |

## Height Bins (All pixels)
| Bin | Arm A MAE | Arm B MAE | Arm A Bias | Arm B Bias |
|---|---|---|---|---|
| 0.0-2.0 | 0.358 | 0.116 | 0.315 | 0.085 |
| 2.0-5.0 | 3.793 | 3.565 | -2.562 | -3.139 |
| 5.0-10.0 | 7.173 | 7.251 | -6.426 | -6.962 |
| 10.0-15.0 | 12.249 | 12.569 | -12.183 | -12.558 |
| 15.0-20.0 | 16.700 | 17.139 | -16.674 | -17.138 |
| 20.0-30.0 | 22.107 | 22.903 | -22.040 | -22.892 |
| 30.0-40.0 | 30.748 | 32.646 | -30.611 | -32.629 |
| 40.0-inf | 50.997 | 52.605 | -50.992 | -52.605 |
