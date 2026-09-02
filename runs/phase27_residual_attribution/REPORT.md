# Phase 27 — DEM Residual Attribution & Tall-Height Gap Analysis

This report presents a detailed analysis of what the monocular relative-depth residual contributes to the coarse DEM surface, specifically focusing on the skyscraper scale gap (>40m structures) in unseen New York.

---

## 1. Error Decomposition by Height Regime (New York)

| Height Regime | Pixel Count | True Mean Height | Coarse DEM Mean | DEM MAE | DEM RMSE | DEM Bias | Residual Mean (C) | Residual MAE (C) | Residual RMSE (C) | Residual Bias (C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `non_building` | 16004264 | 0.02m | 6.24m | 6.23m | 9.12m | 6.23m | 5.81m | 5.96m | 8.80m | 5.80m |
| `bldg_all` | 12307288 | 28.57m | 20.47m | 9.55m | 12.51m | -8.10m | 21.03m | 9.03m | 11.95m | -7.54m |
| `bldg_lt_10` | 999790 | 6.22m | 11.18m | 6.12m | 9.06m | 4.96m | 11.15m | 6.14m | 9.09m | 4.93m |
| `bldg_10_20` | 3770587 | 15.77m | 11.85m | 5.09m | 6.11m | -3.92m | 12.05m | 4.95m | 6.03m | -3.72m |
| `bldg_20_30` | 3055252 | 23.72m | 16.89m | 7.34m | 8.42m | -6.83m | 17.32m | 6.97m | 8.12m | -6.40m |
| `bldg_30_40` | 1590220 | 35.52m | 24.03m | 11.79m | 13.48m | -11.48m | 25.37m | 10.50m | 12.14m | -10.14m |
| `bldg_gt_40` | 2891439 | 54.30m | 36.76m | 17.66m | 20.36m | -17.54m | 37.69m | 16.74m | 19.52m | -16.61m |

---

## 2. Skyscraper Gap Analysis (>40m Structures in New York)

*   **True Mean Height:** `54.30m`
*   **Coarse DEM Mean:** `36.76m` (Coarse DEM underestimates skyscrapers by `17.54m` due to spatial resolution pooling/smoothing).
*   **Residual-Enhanced Mean (C):** `37.69m` (Remaining Gap: `16.61m`).
*   **Height Recovered by AI Residual:** `0.93m` (**5.31%** of the missing height is successfully recovered).

---

## 3. Residual Magnitude & Range Analysis

| Target Pixels | Mean Residual | Median Residual | P95 Residual | P99 Residual | Max Residual | Min Residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `all_pixels` | 0.00m | -0.08m | 2.31m | 5.69m | 31.46m | -20.75m |
| `building_pixels` | 0.56m | 0.17m | 3.51m | 7.43m | 31.46m | -20.75m |
| `gt_30_pixels` | 1.08m | 0.43m | 5.27m | 9.36m | 31.46m | -20.75m |
| `gt_40_pixels` | 0.93m | 0.44m | 4.12m | 7.99m | 31.46m | -20.75m |

---

## 4. Spatial Correlation between Residual and True Height

*   **All Building Pixels:** Pearson R = `0.2057` | Spearman R = `0.2627`
*   **Skyscrapers (>30m):** Pearson R = `-0.0407` | Spearman R = `0.0074`
*   **Skyscrapers (>40m):** Pearson R = `0.0037` | Spearman R = `-0.0361`

### Question: Does the residual increase systematically as true height increases?
**No.** The correlation values are extremely close to zero (or even slightly negative). This indicates that the AI relative-depth residual **does not systematically scale up** for taller buildings. The residual acts as a local sharpening factor that adds object-level variation around the coarse DEM base, but it does not carry absolute metric height corrections.

---

## 5. Formulation C vs Formulation D Comparison (New York)

| Formulation | Overall MAE | Building MAE | Skyscraper (>30m) MAE | Skyscraper (>40m) MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Formulation C (DEM + Residual)** | `7.30m` | `9.03m` | `14.52m` | `16.74m` |
| **Formulation D (DEM + Resid + Bldg Mask)** | `7.56m` | `9.31m` | `15.02m` | `17.28m` |

**Verdict on Formulation D:**  
Formulation D (incorporating the U-Net building mask constraint) performs **worse** than Formulation C across all metrics (e.g., Building MAE is `9.31m` vs `9.03m`). This is because errors in the predicted building footprint mask (such as missed buildings or incorrect borders) cause the high-frequency residual detail to be completely zeroed out in valid building areas. Therefore, **Formulation D should be removed** from the final prototype, and we should default to the simpler and globally superior **Formulation C**.

---

## 6. Scientific Answers to Audit Questions

1.  **Is the AI residual primarily sharpening spatial structure, or is it actually recovering missing metric building height?**  
    **The residual is mostly sharpening.** The residual recovered only `0.93m` (5.31%) of the missing skyscraper height gap, and its correlation with true skyscraper height is near zero. The residual recovers high-frequency spatial boundaries (roof slopes, boundaries) but does not predict absolute height variations.
2.  **Why does the DEM-only baseline underestimate skyscrapers?**  
    The 30m downsampling of the ground truth simulates the spatial resolution pooling of satellite radar (SRTM). Individual tall skyscraper structures are spatially smoothed and averaged with lower neighboring pixels, causing the coarse DEM to underestimate the peak elevations by `17.54m`.
3.  **What is the recommended next step?**  
    Since the residual acts primarily as a high-frequency sharpening filter, and the absolute vertical scale is anchored by the coarse DEM, the primary limitation is the spatial resolution smoothing of the DEM. We should transition from simple global scaling to a **building-aware multi-scale alignment module** or improve the **DEM-image alignment** to map the monocular shapes directly onto the local DEM peaks.
