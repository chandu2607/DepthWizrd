# Phase 29 Peak-Recovery Network Design Specifications

This document locks all design parameters, feature indices, model definitions, and equations before implementing the building peak-recovery model.

---

## 1. Locked 18 Building-Level Features

We strictly utilize exactly 18 features, grouped into three categories:

### A. Coarse DEM Features
1.  **`dem_mean`**: Mean coarse elevation over the building footprint.
2.  **`dem_median`**: Median coarse elevation over the building footprint.
3.  **`dem_p95`**: 95th percentile of coarse elevation over the building footprint.
4.  **`dem_range`**: Local range (max - min) of coarse elevation over the building footprint.
5.  **`dem_std`**: Standard deviation of coarse elevation over the building footprint.

### B. Building footprint Geometry
6.  **`area`**: Total building footprint area in pixels.
7.  **`w_box`**: Bounding box width of the building footprint in pixels.
8.  **`h_box`**: Bounding box height of the building footprint in pixels.
9.  **`aspect_ratio`**: Aspect ratio (width / height) of the bounding box.
10. **`perimeter`**: Perimeter length of the footprint contour in pixels.
11. **`compactness`**: Shape compactness ratio ($perimeter^2 / (4 \pi \cdot area)$).

### C. Local Relative-Depth Features
12. **`d_mean`**: Mean relative depth over the building footprint.
13. **`d_median`**: Median relative depth over the building footprint.
14. **`d_p90`**: 90th percentile of relative depth over the building footprint.
15. **`d_p95`**: 95th percentile of relative depth over the building footprint.
16. **`d_p99`**: 99th percentile of relative depth over the building footprint.
17. **`d_std`**: Standard deviation of relative depth over the building footprint.
18. **`d_range`**: Difference between maximum and minimum relative depth over the footprint.

---

## 2. Frozen Footprint Model Checkpoint

The predicted building mask is extracted using the frozen U-Net footprint model from Phase 24:
*   **Checkpoint Path:** `runs/phase24_moe/seed_0/model.pt`
*   **Model Architecture:** `SmallFusionUNet` backbone within `BuildingConditionedHeightNet`.
*   **Checkpoint Identity:** Best validation checkpoint on Copenhagen (Seed 0).
*   **Footprint Mask Threshold:** Sigmoid probability $>0.5$ (`probs > 0.5`).
*   **Input Resolution:** `256 x 256` resized back to `512 x 512` for full-resolution building extraction.

---

## 3. Coarse DEM Semantics & Delta_H Definition

*   **Coarse DEM Quantity ($Z_{coarse}$):** Coarse nDSM (normalized Digital Surface Model representing above-ground heights in meters).
*   **Source:** Derived by downsampling the ground truth nDSM (`gt`) by a factor of 30 using block average pooling.
*   **Target height ($H_{true}$):** $H_{true\_P95}$ (95th percentile of the true nDSM values inside the building footprint).
*   **Building-height estimate ($H_{coarse}$):** Mean of upsampled coarse DEM (`dem_up`) over the predicted building mask.
*   **Delta Height ($\Delta H$):** 
    $$\Delta H = H_{true\_P95} - H_{coarse}$$
    Since both $H_{true}$ and $H_{coarse}$ represent above-ground heights in meters, this subtraction is dimensionally and semantically fully valid.

---

## 4. Model Architecture & Loss

### PyTorch PeakRecoveryMLP:
```
  [18 Normalized Features] ──> Linear(18, 64) ──> ReLU() ──> Linear(64, 64) ──> ReLU() ──> Linear(64, 1) ──> [ΔH_pred]
```
*   **Loss Function:** Smooth L1 (Huber) Loss.
*   **Output Range:** Continuous real number (supports negative, zero, and positive corrections). No exponential layers are used.

---

## 5. Reconstruction Formula

*   **Building Height Prediction:**
    $$H_{pred} = H_{coarse} + \Delta H_{pred}$$
*   **Dense nDSM Prediction:**
    $$nDSM_{pred}(x,y) = dem\_up(x,y) + \sum_{i} \text{mask}_i(x,y) \cdot \Delta H_{pred, i}$$
    where $i$ iterates over all connected component footprints.
*   **Dense DSM Prediction (if DTM available):**
    $$DSM_{pred}(x,y) = nDSM_{pred}(x,y) + DTM(x,y)$$
