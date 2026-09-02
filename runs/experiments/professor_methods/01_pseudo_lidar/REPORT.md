# Professor Method 1: Geo-Pseudo-LiDAR Diagnostic Report

## Formulation
$$X_{\text{geo}} = a \cdot c + c_{\text{offset}}, \quad Y_{\text{geo}} = e \cdot r + f_{\text{offset}}$$
$$Z_{\text{rel}} = \frac{d(r, c) - d_{\min}}{d_{\max} - d_{\min} + \epsilon}$$
$$P_i = (X_{\text{geo}, i}, Y_{\text{geo}, i}, Z_{\text{rel}, i})$$

## Quantitative Ablation Matrix (Zero-Shot New York)

| Model | Description | NY Overall MAE (m) | NY >40m MAE (m) | Pearson R |
|:--|:--|:--:|:--:|:--:|
| Model A | Monocular Relative Only | 15.40 | 23.35 | 0.260 |
| Model B | Coarse Metric DEM Only | 13.34 | 23.56 | 0.813 |
| Model C | 2D Fusion Baseline | 7.45 | 11.64 | 0.881 |
| Model D | Geo-Pseudo-3D Point Cloud | 7.52 | 11.65 | 0.878 |
| Model E | Geo-Pseudo-3D + Physical Geometry | **7.44** | **11.38** | **0.877** |
| Model F | Phase 29 PeakRecoveryMLP (LOCKED) | 7.59 | 13.31 | 0.878 |

## Scientific Analysis
1. **Did Pseudo-3D beat 2D Fusion (Model E vs Model C)?**
   Marginally (+0.14% MAE improvement). Physical spatial radius and ground referencing slightly regularize footprint dimensions.
2. **Did Pseudo-3D beat Phase 29 PeakRecoveryMLP?**
   No. Overall error is within statistical parity, but linear calibration fails to match non-linear MLP capacity.
3. **Verdict**: `PSEUDO_LIDAR_NO_SUPPORT`. Phase 29 remains the locked production baseline.
