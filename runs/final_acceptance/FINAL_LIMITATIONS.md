# DepthWizard Final Limitations Report

## 1. Scientifically Honest Limitations

### 1.1 Monocular Scale Ambiguity (FUNDAMENTAL)
Single optical RGB images do not contain absolute metric scale. The formula:  
$$Z_{\text{metric}} = f(d_{\text{relative}})$$  
requires at least one of: coarse DEM/SRTM, GCPs, or structural priors to resolve scale.  
Without them, all output is reported as `RELATIVE (rDSM)`.

### 1.2 Tall Building Height Recovery
Phase 29 PeakRecoveryMLP achieves ~44.81% recovery on buildings >40m. Supertall structures (>100m) exhibit monocular perspective saturation. Absolute height accuracy for structures >80m requires LiDAR or stereo anchoring.

### 1.3 DFC2023 Landscape Coverage
The training and evaluation dataset (DFC2023) contains 11 cities across urban environments. Forested and sparse rural landscapes are under-represented. Performance on dense vegetation canopies is not validated.

### 1.4 GSD Dependency
Sub-building-footprint accuracy requires optical imagery with GSD < 1.5m. Coarser imagery degrades footprint segmentation and height localization.

### 1.5 Phase 35 Incomplete Report
Phase 35 streaming sparse anchor evaluation completed all numerical evaluations but the final REPORT.md generation failed at a Python f-string formatting step (`ValueError: Invalid format specifier`). The core quantitative results exist in intermediate CSV tables but the narrative verdict was not written automatically. Verdict: `SPARSE_METRIC_PARTIAL_SUPPORT` (documented manually).

## 2. Technical Boundaries

| Feature | Status | Boundary |
|:--|:--:|:--|
| Absolute metric DSM | Available when GeoTIFF + coarse DEM reference | PNG/JPG always relative |
| Interactive WebGL 3D Flythrough | Available in all modern browsers | Requires WebGL 2.0 support |
| Building height measurement | Per-structure P95 roof vs median DTM ground | ±5-10m uncertainty on tall structures |
| Slope analysis | Available for all tiles | Mean terrain slope inflated near building edges |
| Multi-view geometry | Optional extension only | SIH problem is inherently single-view |
| Real sparse LiDAR fusion | Not available (no physical sensor) | Phase 35 uses simulated ground-truth sampling as proxy |
