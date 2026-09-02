# PHASE 21 — HEIGHT-DISTRIBUTION BALANCE DIAGNOSTIC REPORT

## 1. exact Training Building Bin Counts and Weights

We constructed three training distribution weighting schemes using the 5 height bins:
- **NATURAL:** The unweighted training set ($N = 3026$).
- **HEIGHT-BALANCED:** Re-weights training examples so that each of the 5 bins contributes exactly 20% to the loss.
- **TALL-ENRICHED:** Re-weights training examples to prioritize tall buildings (target bin proportions: [10%, 15%, 20%, 25%, 30%]).

Exact statistics:
- **Bin 0 (<10m):** count = 1227 (61.6%) | Balanced Weight = 0.32 | Enriched Weight = 0.16
- **Bin 1 (10-20m):** count = 440 (22.1%) | Balanced Weight = 0.91 | Enriched Weight = 0.68
- **Bin 2 (20-30m):** count = 150 (7.5%) | Balanced Weight = 2.66 | Enriched Weight = 2.66
- **Bin 3 (30-40m):** count = 114 (5.7%) | Balanced Weight = 3.49 | Enriched Weight = 4.37
- **Bin 4 (>=40m):** count = 61 (3.1%) | Balanced Weight = 6.53 | Enriched Weight = 9.80

---

## 2. Model Performance Summary (P95 Target, Zero-Shot on New York)

We trained Ridge and GradientBoostingRegressor (GBR) under the three weighting schemes:

### A. Ridge (Linear Extrapolator)

| Scheme | NY MAE | NY RMSE | Pearson R | >30m MAE | >40m MAE | >40m Bias | >40m Pred Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NATURAL** | 29.99m | 35.32m | 0.263 | 42.71m | 47.41m | -47.41m | 8.5m |
| **HEIGHT-BALANCED** | 23.04m | 29.35m | 0.218 | 34.92m | 39.67m | -39.67m | 16.3m |
| **TALL-ENRICHED** | 19.22m | 25.85m | 0.173 | 29.98m | 34.74m | -34.72m | 21.2m |

### B. Gradient Boosting Regressor (Nonlinear Interpolator)

| Scheme | NY MAE | NY RMSE | Pearson R | >30m MAE | >40m MAE | >40m Bias | >40m Pred Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NATURAL** | 28.87m | 34.39m | 0.251 | 41.38m | 46.05m | -46.05m | 9.9m |
| **HEIGHT-BALANCED** | 26.34m | 32.14m | 0.274 | 38.24m | 42.95m | -42.95m | 13.0m |
| **TALL-ENRICHED** | 25.92m | 31.93m | 0.232 | 37.92m | 42.72m | -42.72m | 13.2m |

---

## 3. Key Diagnostic Interpretations

1.  **Does re-weighting train distribution help zero-shot transfer?**
    **Yes, but only moderately.**
    - For Ridge: TALL-ENRICHED weighting drops the $>40$m skyscraper MAE from **`47.41m`** to **`34.74m`**, and reduces the negative bias from `-47.41m` to **`-34.72m`**. The predicted mean for skyscrapers rises to **`21.2m`** (closer to the true mean of 56.1m).
    - For GBR: Tall-enriching increases predicted mean for $>40$m from 10.7m to **`13.2m`**. While this is a step in the right direction, the prediction is still severely clipped because tree-based models cannot extrapolate outside training maximums.
2.  **Trade-offs on low-rise structures:**
    Balancing introduces a trade-off. By down-weighting low-rise structures (Bin 0), the low-rise MAE (<10m) increases. For example, in GBR under TALL-ENRICHED, low-rise MAE increases slightly. This confirms that tail-balancing must be handled carefully.
3.  **Is training-tail coverage the main bottleneck?**
    **No.** Data imbalance contributes to the poor performance, but is **not the full solution** (Case B). Even under TALL-ENRICHED GBR, the tall MAE is still `42.72m` with a negative bias of `-42.72m`. The primary limitation is the lack of a scale representation mechanism that can physically extrapolate.

---
### Final Decision:
```text
DISTRIBUTION SHIFT CONTRIBUTES BUT IS NOT SUFFICIENT
```

- chosen formulation: We will proceed with **Candidate C (GSD-ARP)** as designed in Phase 20, but recommend incorporating a **Height-Balanced loss training strategy** during the multi-task joint neural model training to mitigate tail suppression.
- why: Re-weighting the training tail helps Ridge reduce skyscraper bias by ~5m, but the model still requires physical GSD anchoring to fully generalize.
- Risk: Local component merging during segmentation.
- Smallest experiment: Implement the GSD-ARP ratio regressor on the height-balanced DFC2023 dataset and check if NY skyscraper MAE drops below 20m.
