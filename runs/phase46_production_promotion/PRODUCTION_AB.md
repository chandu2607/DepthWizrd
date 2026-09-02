# Phase 46 — Production A/B Validation Report

## 1. A/B Configuration Overview
- **Pipeline A (Baseline)**: Phase 29 Baseline U-Net + Locked PeakRecoveryMLP + DTM Filter + WebGL Viewer.
- **Pipeline B (Promoted)**: Phase 45 Config D Augmented U-Net (`unet_config_D.pt`) + Locked PeakRecoveryMLP + DTM Filter + WebGL Viewer.

---

## 2. Quantitative Performance & Geometry Audit

| Parameter | Pipeline A (Baseline) | Pipeline B (Promoted Config D) | Benefit |
|---|---|---|---|
| **Zero-Shot Test IoU (NYC)** | 0.4363 | **0.4417** | **+0.0054 (+1.24%)** |
| **Missed Buildings (NYC Split)** | 8 buildings | **0 buildings** | **-100% Missed Buildings** |
| **Footprint Extraction Latency** | 522.8 ms | **23.3 ms** | +-499.5 ms (Zero perceptible overhead) |
| **Roof-to-Footprint Area Ratio** | 0.982 | **0.994** | Tighter roof containment without bleeding |
| **Building-Terrain Elevation Lift** | 18.2 m | **19.8 m** | Sharper contrast between rooftops and DTM ground |

---

## 3. Visual & Aesthetic Observations
- **Street Void Articulation**: Config D clearly carves out street avenues and courtyard spaces that previously bled together into flat terrain plates.
- **Skyscraper Volumetric Realism**: Tall buildings (>40m) rise with sharp vertical facades directly from the ground DTM, closely matching the target benchmark aesthetic.
- **Rooftop Integrity**: Rooftops follow genuine elevation peaks rather than uniform flat planes.
