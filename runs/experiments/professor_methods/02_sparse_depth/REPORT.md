# Professor Method 2: Sparse Depth Completion Diagnostic Report

## Formulation
Simulated sparse metric anchors $S(x, y)$ sampled from ground-truth elevation at density $p \in [0.0001, 0.05]$:
$$S(x, y) = Z_{\text{true}}(x, y) \quad \text{at sparse indices}$$
$$\hat{Z}_{\text{dense}}(x, y) = \alpha (a \cdot d_{\text{norm}}(x, y) + b) + (1 - \alpha) \text{DEM}_{\text{coarse}}(x, y)$$
where $(a, b)$ are fitted via Huber regression on observed sparse anchors.

## Model Comparison at 0.5% Anchor Density (New York Zero-Shot)

| Method | NY Overall MAE (m) | NY >40m Skyscraper MAE (m) | Note |
|:--|:--:|:--:|:--|
| Baseline A: Coarse DEM Only | 13.34 | 23.56 | 0% metric anchors |
| Model D: Depth-Guided Affine (Sparse Only) | 8.12 | 12.85 | No coarse DEM |
| Model E: Sparse + Monocular + Coarse Blend | **7.38** | **11.20** | 50/50 affine + coarse blend |
| Model F: Phase 29 PeakRecoveryMLP (LOCKED) | 7.63 | 13.36 | Production baseline |

## Key Findings
1. **Real Metric Grounding**: Genuine metric anchors provide real scale calibration (unlike monocular pseudo-points).
2. **Density Sensitivity**: A minimum density of $\ge 0.1\%$ (approx. 260 points per $512 \times 512$ tile) is required for reliable metric anchoring.
3. **Noise Sensitivity**: Affine depth guidance remains robust to $\pm 0.25\text{m}$ sensor noise.
4. **Verdict**: `SPARSE_METRIC_PARTIAL_SUPPORT` -- effective as a complementary sensor fusion pathway when sparse LiDAR/radar is available.
