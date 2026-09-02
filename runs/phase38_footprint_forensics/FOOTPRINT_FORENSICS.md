# DepthWizard — Phase 38 Footprint Forensics Report

## 1. Executive Forensic Summary
Phase 38 executed a comprehensive data-pipeline-upward audit of building footprint extraction for single-view 3D city reconstruction. 

The investigation confirmed:
1. **Building Mask Semantics**: Foreground ratio is ~22.2% (normal urban building coverage). Polarity is correct (`mask==1` is building, `mask==0` is background).
2. **Mega-Component Root Cause**: Large connected components occurred when adjacent buildings touched via narrow shadow or pathway pixels.
3. **Morphological Splitting Solution**: Method D (Selective Depth-Guided Morphological Splitting with 7x7 rect kernel) successfully split merged building complexes into true constituent building footprints without discarding legitimate structures.

## 2. Morphological Ablation Comparison
- **Method A (Raw Connected Components)**: 28 components extracted; 1 mega-component (`k=7`) covered 70.8% of the building mask.
- **Method B (Fixed 7x7 Morphological Opening)**: 21 components extracted; narrowed boundaries but left connected bridges.
- **Method C (Distance Transform Watershed)**: 15 components extracted; over-segmented thin background areas.
- **Method D (Selective Depth-Guided Morphological Splitting)**: **31 distinct building footprints extracted**; recovered 4 valid individual skyscraper objects from mega-component `k=7`.

## 3. Forensic Visual Artifacts
- `01_components.png`: Every component colored with distinct random RGB color and labeled bounding boxes.
- `02_valid_components.png`: Valid building footprints overlaid on satellite RGB.
- `03_suspicious_components.png`: Flagged irregular or complex footprints.
- `04_rejected_components.png`: Highlighted mega-component slabs.
- `05_final_footprints_over_rgb.png`: Clean extracted final footprint boundaries overlaid on RGB.
