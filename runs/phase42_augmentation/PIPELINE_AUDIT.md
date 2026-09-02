# PHASE 42 PIPELINE AUDIT

## 1. Dataset Splits
- **Dataset Path**: \data/dfc2023_multicity- **Manifest**: uns/dfc2023_multicity_prep/split_manifest.csv- **Train Cities**: Barcelona, Berlin, Brasilia, NewDelhi, Portsmouth, Rio, SanDiego, SaoLuis, Sydney
- **Validation City**: Copenhagen (Used for Model Selection)
- **Test City**: NewYork (Zero-Shot)

## 2. Model 1: Building Footprint Learning
- **Model**: \BuildingConditionedEstimator\ (U-Net)
- **Responsibility**: Predicts building footprints/masks from RGB+Depth.
- **Image Dimensions**: Typically trained at 256x256 (Phase 24 used 256).
- **Loss**: \F.binary_cross_entropy_with_logits\ (footprint mask).
- **Optimizer**: Adam, lr=1e-3.

## 3. Model 2: Relative Depth
- **Model**: \DepthAnythingV2- **Responsibility**: Provides dense structural cues (frozen, inference-only).
- **Inference Res**: Target HW of original RGB (varies by tile).

## 4. Model 3: Peak Recovery (Phase 29 Baseline)
- **Model**: \PeakRecoveryMLP\ (4-layer MLP: 18 -> 64 -> 64 -> 1).
- **Responsibility**: Predicts $\Delta H$ for each building instance to recover the true peak (P95).
- **Input Features**: 18 tabular features (dem_mean, dem_median, dem_p95, dem_range, dem_std, area, w_box, h_box, aspect_ratio, perimeter, compactness, d_mean, d_median, d_p90, d_p95, d_p99, d_std, d_range).
- **Target**: \	rue_p95 - dem_mean- **Normalization**: \(X - mu_train) / (sigma_train + 1e-6)- **Loss**: Weighted Huber Loss (\F.huber_loss\), weighted by true building height bin (<10m: 1.0, 10-20m: 1.5, 20-30m: 2.0, 30-40m: 2.5, >=40m: 3.0).
- **Optimizer**: Adam, lr=5e-3.
- **Epochs**: 120.
- **Seeds**: 0, 1.
- **Checkpoint Selection Rule**: Best validation MAE on Copenhagen reconstructed height (\pred_recon_val - true_p95\).
