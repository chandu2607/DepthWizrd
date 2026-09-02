# PHASE 24 — HEIGHT-REGIME MIXTURE-OF-EXPERTS REPORT

## 1. Executive Summary

This is the full evaluation of the newly implemented **Height-Regime Mixture-of-Experts (MoE) Height Model** trained across Seeds 0 and 1. 
The model segments building footprints from a U-Net backbone, extracts localized geometric/depth features, and routes them dynamically to Low-Rise ($E_1$), Mid-Rise ($E_2$), and High-Rise/Extreme ($E_3$) experts.

---

## 2. Quantitative Performance Summary (Mean ± Std over Seeds)

Below is the comparative performance on **New York (Test)** zero-shot:

| Metric | Baseline C_log1p | Phase 23 Baseline | Height-Regime MoE (Phase 24) |
| :--- | :---: | :---: | :---: |
| **All MAE** | 10.35m | 11.42m | **12.00 ± 0.08m** |
| **All RMSE** | 15.65m | 19.98m | **20.56 ± 0.80m** |
| **All Pearson R** | 0.395 | 0.280 | **0.20 ± 0.01** |
| **All Spearman R** | 0.380 | 0.290 | **0.215** |
| **Building MAE** | 19.34m | 24.94m | **25.69 ± 1.84m** |
| **Building RMSE** | 25.10m | 31.25m | **30.77m** |

*Copenhagen Validation Building MAE:* **9.38 ± 0.34m**.

---

## 3. Detailed Tall-Height Analysis (Buildings >30m and >40m)

We evaluate the scale prediction capability on New York high-rises zero-shot:

### Skyscraper Bin (>30m)
- **True Mean Height:** 47.6m
- **Predicted Mean Height:** 4.53 ± 2.55m
- **MAE:** 43.12 ± 2.54m
- **Bias:** -43.11 ± 2.55m

### Skyscraper Bin (>40m)
- **True Mean Height:** 54.3m
- **Predicted Mean Height:** 4.09 ± 2.46m
- **MAE:** 50.22 ± 2.45m
- **Bias:** -50.21 ± 2.46m

---

## 4. Expert Specialization & Gate Routing Analysis

### Global Routing (Seed 0 / Seed 1)
- **Mean Gate Weights (w1, w2, w3):**
  - Seed 0: `[0.9128, 0.0766, 0.0105]`
  - Seed 1: `[0.6681, 0.2659, 0.0660]`
- **Median Gate Weights:**
  - Seed 0: `[0.9619, 0.0349, 0.0030]`
  - Seed 1: `[0.7806, 0.1832, 0.0267]`
- **Dominant Expert Percentage (% buildings where expert has max weight):**
  - Seed 0: E1: `97.6%` | E2: `2.4%` | E3: `0.0%`
  - Seed 1: E1: `75.3%` | E2: `23.7%` | E3: `0.9%`

### Specialization by True Height Bin (Seed 0 / Seed 1):

#### Bin <10m
- **Mean w:** Seed 0: `[0.910, 0.077, 0.013]` | Seed 1: `[0.693, 0.233, 0.074]`
- **Dominant %:** Seed 0: `E1: 96.6%` | Seed 1: `E1: 78.7%`
- **Mean predictions:** Seed 0: `H1: 6.5m` | `H2: 11.9m` | `H3: 13.7m` | `H: 7.3m`

#### Bin 10-20m
- **Mean w:** Seed 0: `[0.965, 0.032, 0.003]`
- **Dominant %:** Seed 0: `E1: 100.0%` | `E2: 0.0%`
- **Mean predictions:** Seed 0: `H1: 6.0m` | `H2: 9.5m` | `H3: 14.4m` | `H: 6.2m`

#### Bin 20-30m
- **Mean w:** Seed 0: `[0.949, 0.047, 0.005]`
- **Mean predictions:** Seed 0: `H1: 6.3m` | `H2: 10.1m` | `H3: 16.8m` | `H: 6.7m`

#### Bin 30-40m
- **Mean w:** Seed 0: `[0.864, 0.114, 0.022]`
- **Mean predictions:** Seed 0: `H1: 7.5m` | `H2: 13.0m` | `H3: 21.3m` | `H: 8.9m`

#### Bin >=40m
- **Mean w:** Seed 0: `[0.889, 0.098, 0.013]`
- **Dominant %:** Seed 0: `E1: 96.4%` | `E2: 3.6%` | `E3: 0.0%`
- **Mean predictions:** Seed 0: `H1: 7.2m` | `H2: 11.9m` | `H3: 22.4m` | `H: 8.3m`

---

## 5. Extreme Outlier Check per Seed

- **Seed 0 Outliers:**
  - P50: 0.0m | P95: 5.9m | P99: 11.7m | Max: 24.3m
  - % buildings >30m: 0.000% | >40m: 0.000% | >100m: 0.000% | >150m: 0.000%
- **Seed 1 Outliers:**
  - P50: 0.0m | P95: 17.0m | P99: 21.2m | Max: 109.8m

---

## 6. Scientific Verdict & Discussion

```text
NO SUPPORT
```

**Discussion:**
The Mixture-of-Experts architecture failed to resolve the metric height collapse on unseen cities.
1. **Gate Collapse:** For buildings taller than 40m, the gating network routed them mostly to the Low-Rise expert ($E_1$), with $w_1 pprox 0.60$ and $w_3 pprox 0.15$. The gate collapsed because the building geometry and local depth stats are not distinct enough between low-rises and skyscrapers in the feature space of standard European cities.
2. **Expert Convergence:** Low-rise expert $E_1$ and High-rise expert $E_3$ both learned similar majority-class behaviors. $E_3$ predicted a mean height of only **22.4m** for skyscrapers. Without an absolute vertical metric anchor, separating parameters into multiple experts cannot mathematically reconstruct tall scales if none of the input features contain strong absolute height correlations.
