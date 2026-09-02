# Phase 77 in-distribution terrain learning control

## Split
- Data source: frozen Phase 72 Uttarakhand grid only.
- Deterministic split: upper-left valid extent for training, lower-right valid extent for validation, with no overlap.
- Train bbox: (1774, 2513, 2286, 3025)
- Validation bbox: (5834, 7335, 6346, 7847)

## Target proof
- Train crop local median: 5240.126953 m.
- Train crop local relief mean: -31.671883 m, std: 380.598511 m.
- Validation crop local median: 5000.000000 m.
- Validation crop local relief mean: -53.076824 m, std: 561.360291 m.

## One-epoch held-out Uttarakhand validation
- MAE: 463.506775 m
- RMSE: 570.499573 m
- Pearson: 0.064182 if finite else NaN
- Mean bias: 102.204666 m
- Prediction std: 1.436009 m
- Target std: 561.360291 m
- Error std: 561.269958 m
- Prediction min/max/mean: 46.879311 / 54.584412 / 49.127838 m
- Target min/max/mean: -1327.232178 / 1185.815430 / -53.076824 m
- Gradient-x MAE: 6.183001 m
- Gradient-y MAE: 4.136543 m
- Mean gradient MAE: 5.159772 m

## Baselines on the same held-out crop
- Zero prediction: MAE=462.019531 m, RMSE=563.863953 m, Pearson=nan
- Training mean prediction: MAE=462.716492 m, RMSE=561.768250 m, Pearson=nan
- Training median prediction: MAE=462.019562 m, RMSE=563.864014 m, Pearson=nan

## Cross-region comparison
- Cross-region Phase 75: MAE=49.204597 m, RMSE=49.247993 m, Pearson=0.035733
- In-distribution Phase 77: MAE=463.506775 m, RMSE=570.499573 m, Pearson=0.0641815142515434

## Decision
IN_DISTRIBUTION_DIAGNOSIS_INCONCLUSIVE

